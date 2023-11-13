import uuid
import json
from tqdm.auto import tqdm
import torch
import time
from PIL import Image
import torch.multiprocessing as multiprocessing
from typing import Any, Callable, Dict, List, Optional, Union

from block_stage_file import *

torch.set_grad_enabled(False)

def set_progress_bar(iterable=None, total=None):
    _progress_bar_config = {}
    if iterable is not None:
        return tqdm(iterable, **_progress_bar_config)
    elif total is not None:
        return tqdm(total=total, **_progress_bar_config)
    else:
        raise ValueError("Either `total` or `iterable` has to be defined.")

def clip_worker(clip_queue, unet_queue, device):
    # initialize the model
    clip_stage = clip(device=device)

    while True:
        # batch_input can not be empty
        # get the first batch item
        item = clip_queue.get()
        batch_input = [item]

        # batching part
        while not clip_queue.empty():
            batch_input.append(clip_queue.get())

        """item = clip_queue.get()
        if item is None:
            # terminal tag
            unet_queue.put(None)
            break"""
        #print(f"CLIP processing... uuid: {item['uuid']}")
        
        print(f"CLIP processing batch size: {len(batch_input)}, id: {[item['id'] for item in batch_input]}")

        prompts = []
        for input in batch_input:
            prompts.append(input["prompt"])

        prompt_embeds = clip_stage.encode_prompt(prompts)
        """
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        所以每一个prompt对应的两个embeds(do_classifier_free_guidance = True), 其索引是分开的,
        """
        batch_length = len(batch_input)
        for idx in range(batch_length):
            output = {
                "origin_workload": batch_input[idx],
                "negative_prompt_embeds": prompt_embeds[idx],
                "prompt_embeds": prompt_embeds[idx + batch_length],
                # latent intialization put to unet_worker
            }
            unet_queue.put(output)
        del prompt_embeds
        torch.cuda.empty_cache()

def unet_worker(unet_queue, vae_queue, device, batch_size_config, delay_num, image_size, log_file, msg_queue):
    from test_set import profile_latency
    with open("detailed_batch_profile_latency.json", "r") as json_file:
        detailed_batch_profile_latency = json.load(json_file)

    unet_stage = unet(device=device)

    begin_time = 0
    batch_count = 0
    if log_file:
        file_name = f"unet_batch_iteration_SLOs_aware_delay_{delay_num}.log"
        unet_batch_log_file = open(file_name, "w")
    # 测试了写入3个字节的用时为：4.291534423828125e-06
    # 所以只是记录batch的信息应该不会太影响性能

    batch_info_collection = {} # 为了画箱线图
    
    # PriorityQueue只在unet_worker的进程中使用，所以直接用queue，不需要考虑跨进程安全
    from queue import PriorityQueue
    queue_count = 0
    abandoned_request = 0
    unet_priority_queue_list = {
        256: PriorityQueue(),
        512: PriorityQueue(),
        768: PriorityQueue(),
        1024: PriorityQueue(),
        }

    # 初始化在大循环外，记录目前loop中的batch信息
    batch_input = []
    while True:

        # 把unet_queue中的item都取出来，放入不同size对应的priority_queue里
        while not unet_queue.empty():
            item = unet_queue.get()
            if begin_time == 0:
                begin_time = item["origin_workload"]["request_time"]
            item["negative_prompt_embeds"] = item["negative_prompt_embeds"].to(device)
            item["prompt_embeds"] = item["prompt_embeds"].to(device)
            item["timesteps"], item["sigmas"] = unet_stage.get_timesteps_and_sigmas(item["origin_workload"]["inference_steps"], device=device)
            item["unet_loop_count"] = 0
            item["latent"] = unet_stage.prepare_latents(1, 
                                                        num_inference_steps=item['origin_workload']['inference_steps'],
                                                        height=item['origin_workload']['height'],
                                                        width=item['origin_workload']['width'],
                                                        device=device,
                                                        dtype=unet_stage.dtype
                                                        )
            print(f"unet_queue put item with request id: {item['origin_workload']['id']}""\n""-----------------------\n")
            
            # 先检查能否满足SLOs
            if time.time() + profile_latency[str(image_size)][str(item['origin_workload']['inference_steps'])] > item["origin_workload"]["request_time"] + item["origin_workload"]["SLOs"]:
                unet_queue.get()
                abandoned_request += 1
            else:
                priority = item["origin_workload"]["request_time"] + item["origin_workload"]["SLOs"] - time.time() + profile_latency[str(image_size)][str(item['origin_workload']['inference_steps'])]
                unet_priority_queue_list[image_size].put((priority, item))
                queue_count += 1

        valid_request_list = []
        while not unet_priority_queue_list[image_size].empty():
            item = unet_priority_queue_list[image_size].get()
            if time.time() + profile_latency[str(image_size)][str(item[1]['origin_workload']['inference_steps'])] > item[1]["origin_workload"]["request_time"] + item[1]["origin_workload"]["SLOs"]:
                abandoned_request += 1
            else:
                valid_request_list.append(item)
        for item in valid_request_list:
            unet_priority_queue_list[image_size].put(item)
        # 告诉VAE已经丢弃的request数量
        msg_queue.put(abandoned_request)

        # 要么继续已有的iteration，要么从queue重新构建batch
        if len(batch_input) == 0 and unet_priority_queue_list[image_size].qsize() == 0:
            continue

        queue_size_before_schedule = unet_priority_queue_list[image_size].qsize()
        batch_size_before_schedule = len(batch_input)

        put_back = []
        SLOs_flag = True
        while not unet_priority_queue_list[image_size].empty() and len(batch_input) < batch_size_config[image_size]:
            if not SLOs_flag:
                break
            # batch里必须至少有一个
            if len(batch_input) == 0:
                batch_input.append(unet_priority_queue_list[image_size].get())
                queue_count -= 1    
            else: # 考虑新加入request的case
                new_item = unet_priority_queue_list[image_size].get()
                new_batch_size = len(batch_input) + 1
                current_batch_size = new_batch_size

                # 细粒度的SLO check，只是先写一下
                remain_steps_list = []
                deadline_list = []
                estimate_finish_time_list = [-1] * new_batch_size
                check_flag = [False] * new_batch_size
                temp_time = time.time()
                for item in batch_input:
                    remain_steps_list.append(item[1]["origin_workload"]["inference_steps"] - item[1]["unet_loop_count"])
                    deadline_list.append(item[1]["origin_workload"]["request_time"] + item[1]["origin_workload"]["SLOs"])
                remain_steps_list.append(new_item[1]["origin_workload"]["inference_steps"])
                deadline_list.append(new_item[1]["origin_workload"]["request_time"] + new_item[1]["origin_workload"]["SLOs"])
                while max(remain_steps_list) > 0:
                    if not SLOs_flag:
                        break
                    temp_time += detailed_batch_profile_latency[str(image_size)][str(new_batch_size)]["unet_single_loop"] * 5
                    for idx in range(new_batch_size):
                        remain_steps_list[idx] -= 5
                        if remain_steps_list[idx] <= 0 and check_flag[idx] == False:
                            current_batch_size -= 1
                            check_flag[idx] = True
                            estimate_finish_time_list[idx] = temp_time
                            if estimate_finish_time_list[idx] > deadline_list[idx]:
                                SLOs_flag = False
                                break
                # 细粒度的SLO check，只是先写一下

                # check加入后是否会导致超时
                """for item in batch_input:
                    remain_steps = item[1]["origin_workload"]["inference_steps"] - item[1]["unet_loop_count"]
                    deadline = item[1]["origin_workload"]["request_time"] + item[1]["origin_workload"]["SLOs"]
                    estimate_finish_time = time.time() + detailed_batch_profile_latency[str(image_size)][str(new_batch_size)]["unet_single_loop"] * remain_steps 
                    if deadline < estimate_finish_time:
                        SLOs_flag = False
                    if not SLOs_flag:
                        break"""
                if not SLOs_flag:
                    put_back.append(new_item)
                    break
                else:
                    batch_input.append(new_item)
                    queue_count -= 1
        if len(put_back) != 0:
            unet_priority_queue_list[image_size].put(put_back[0])

        # batch 构建好了
        # 更新一下batch信息
        batch_count += 1
        info = {
            "batch_count": batch_count,
            "passed_time": time.time() - begin_time,
            "batch_size_before_schedule": batch_size_before_schedule,
            "batch_size_after_schedule": len(batch_input),
            "request_id_list": [item[1]["origin_workload"]["id"] for item in batch_input],
            "queue_size_before_schedule": queue_size_before_schedule
            }
        if log_file:
            unet_batch_log_file.write(json.dumps(info) + "\n")

        batch_info_collection[len(batch_input)] = batch_info_collection.setdefault(len(batch_input), 0) + 1
        print(f"yhc debug:: batch_info_collection: {batch_info_collection}\n12580")
        print(f"UNet processing batch size: {len(batch_input)}, size: {batch_input[0][1]['origin_workload']['height']}, id: {[item[1]['origin_workload']['id'] for item in batch_input]}")

        negative_prompt_embeds = []
        prompt_embeds = []
        latents = []
        guidance_scale_list = []
        for input in batch_input:
            negative_prompt_embeds.append(input[1]["negative_prompt_embeds"])
            prompt_embeds.append(input[1]["prompt_embeds"])
            latents.append(input[1]["latent"]) #.to(unet_stage.device)) # 前面已经to device过了
            guidance_scale_list.append(input[1]["origin_workload"]["guidance_scale"])
        prompt_embeds = torch.cat([torch.stack(negative_prompt_embeds), torch.stack(prompt_embeds)]).to(unet_stage.device)
        latents = torch.cat(latents)

        # 用max为保证正确性
        unet_stage.scheduler.set_timesteps(50, device=unet_stage.device)
        timesteps = unet_stage.scheduler.timesteps
        
        """with set_progress_bar(total=5) as progress_bar:
            i = 0
            while i % 5 != 0 or i == 0:
                t = timesteps[i]
                i += 1
                latents = unet_stage.unet_single_loop(prompt_embeds, latents, t, guidance_scale=7.5)
                progress_bar.update()"""
        i = 0
        while i % 5 != 0 or i == 0:
            timesteps_list = []
            sigma_list = []
            sigma_to_list = []
            for item in batch_input:
                timesteps_list.append(item[1]["timesteps"][item[1]["unet_loop_count"]+i:item[1]["unet_loop_count"]+i+1])
                sigma_list.append(item[1]["sigmas"][item[1]["unet_loop_count"]+i:item[1]["unet_loop_count"]+i+1])
                sigma_to_list.append(item[1]["sigmas"][item[1]["unet_loop_count"]+i+1:item[1]["unet_loop_count"]+i+2])
            timesteps_list = torch.cat(timesteps_list)
            sigma_list = torch.cat(sigma_list)
            sigma_to_list = torch.cat(sigma_to_list)

            latents = unet_stage.unet_single_loop_different_timesteps(prompt_embeds, latents, timesteps_list, guidance_scale_list, sigma_list, sigma_to_list)
            i += 1
        print(f"yhc debug:: finish 5 loop with batch_size: {len(latents)}, image size: {latents.shape[3] * 8}")
    
        #print(f"yhc debug:: {batch_input}")
        pop_list = []
        for idx in range(len(batch_input)):
            batch_input[idx][1]["latent"] = latents[idx:idx+1]
            batch_input[idx][1]["unet_loop_count"] += 5
            if batch_input[idx][1]["unet_loop_count"] >= batch_input[idx][1]["origin_workload"]["inference_steps"]:
                pop_list.append(idx)
                output = {
                    "latents": latents[idx:idx+1],
                    "origin_workload": batch_input[idx][1]["origin_workload"],
                    "time_cost": time.time() - batch_input[idx][1]["origin_workload"]["request_time"], # 这好像没用
                    "deadline": batch_input[idx][0]
                }
                vae_queue.put(output)

        # 需要等for循环结束后，再移除已经扔给vae的项
        # 否则会报index error
        updated_batch_input = []
        for idx in range(len(batch_input)):
            if idx not in pop_list:
                updated_batch_input.append(batch_input[idx])
        batch_input = updated_batch_input
        # end 5 unet loop
        # ——————————————————————————————
        torch.cuda.empty_cache()

def vae_worker(unet_queue, vae_queue, total_request, device, batch_config, delay_num, image_size, log_file, msg_queue):
    vae_and_safety_checker_stage = vae_and_safety_checker(device="cuda:0")
    
    if log_file:
        file_name = f"vae_batch_iteration_SLOs_aware_delay_{delay_num}.log"
        f = open(file_name, "w")

    from queue import PriorityQueue
    vae_priority_queue_list = {
        256: PriorityQueue(),
        512: PriorityQueue(),
        768: PriorityQueue(),
        1024: PriorityQueue(),
    }
    queue_count = 0
    image_save_count = 0
    abandoned_request = 0
    request_count = 0
    goodput = 0
    begin_time = 0
    batch_input = []
    while True:
        while not msg_queue.empty():
            abandoned_request = msg_queue.get()
        while not vae_queue.empty():
            item = vae_queue.get()
            if begin_time == 0:
                begin_time = item["origin_workload"]["request_time"]
            vae_priority_queue_list[item["origin_workload"]["height"]].put((item["deadline"], item))
            queue_count += 1

        if len(batch_input) == 0 and queue_count == 0:
            time.sleep(0.1)
            continue

        while not vae_priority_queue_list[image_size].empty():
            batch_input.append(vae_priority_queue_list[image_size].get())
            queue_count -= 1

        latents = []
        for input in batch_input:
            latents.append(input[1]["latents"])
        latents = torch.cat(latents).to(vae_and_safety_checker_stage.device)

        images = vae_and_safety_checker_stage.vae_decode(latents)
        """try:
            images = pipe.decode_latents(latents)
            request_count += len(batch_input)
            print(f"VAE processing batch size: {len(batch_input)}, image_size: {batch_input[0]['origin_workload']['height']}")
        except:
            time.sleep(0.1)
            continue"""
        
        # Run safety checker, result is PIL type
        images, has_nsfw_concept = vae_and_safety_checker_stage.safety_check(images, vae_and_safety_checker_stage.device)
        
        image_save = True
        if image_save:
            for image in images:
                image.save(f"pipeline_result_{image_save_count}.jpg")
                image_save_count += 1
            
        request_count += len(batch_input)
        print(f"VAE processing batch size: {len(batch_input)}, image_size: {batch_input[0][1]['origin_workload']['height']}")

        # count goodput
        for input in batch_input:
            if input[1]["origin_workload"]["request_time"] + input[1]["origin_workload"]["SLOs"] >= time.time():
                goodput += 1
        del latents

        # only if pipe.decode_latents is correctly executed, batch_input will be empty
        batch_input = []

        info = {
            "passed_time": time.time() - begin_time,
            "total_request": total_request,
            "abandoned_request": abandoned_request,
            "finished_request": request_count,
            "goodput:": goodput,
            "SLO_rate": goodput / (request_count + abandoned_request)
        }
        if log_file:
            f.write(json.dumps(info) + "\n")

        #if request_count >= total_request or (unet_queue.empty() and vae_queue.empty()):
        print("!iteration SLOs aware, no preemption")
        print(f"delay_num: {delay_num}")
        print(f"batch_size_config: {batch_config}")
        print(f"total request: {total_request},\nabandoned request: {abandoned_request}, \nfinished request: {request_count},\ntotal goodput: {goodput},\ntotal latency: {time.time() - begin_time},\ntotal throughput: {(total_request / (time.time() - begin_time))}")
        print(f"SLO rate: {round(goodput / (request_count + abandoned_request), 2)}")

        torch.cuda.empty_cache()
        """for idx in range(len(images)):
            images[idx].save(f"{batch_input[idx]['origin_workload']['uuid']}.jpg")
            print(f'uuid: {batch_input[idx]["origin_workload"]["uuid"]}, time_cost: {time.time() - batch_input[idx]["origin_workload"]["request_time"]}')"""

if __name__ == "__main__":
    device = "cuda:5"
    clip_device = "cuda:0"
    unet_device = "cuda:1"
    vae_device = "cuda:2"
    image_size = 512
    batching_policy = "size" # delay, naive

    dynamic_batch_size_config = {
        256: 24,
        512: 15,
        768: 8,
        1024: 5
    }

    torch.multiprocessing.set_start_method("spawn")
    clip_queue = multiprocessing.Queue()
    unet_queue = multiprocessing.Queue()
    vae_queue = multiprocessing.Queue()

    msg_queue = multiprocessing.Queue()

    loop_count = 60
    delay_num = 4.5
    log_file = False
    # Start processes
    processes = [
        multiprocessing.Process(target=clip_worker, args=(clip_queue, unet_queue, clip_device)),
        multiprocessing.Process(target=unet_worker, args=(unet_queue, vae_queue, unet_device, dynamic_batch_size_config, delay_num, image_size, log_file, msg_queue)),
        multiprocessing.Process(target=vae_worker, args=(unet_queue, vae_queue, loop_count, vae_device, dynamic_batch_size_config, delay_num, image_size, log_file, msg_queue))
    ]

    for p in processes:
        p.start()
        #break
    
    # load model to GPU need several seconds
    # so we'd better sleep some seconds before send request
    time.sleep(30)

    from test_set import arrive_interval, steps_list, prompt_list, lora_list, size_list, profile_latency
    begin = time.time()
    for idx in range(loop_count):
        time.sleep(arrive_interval[idx] * delay_num)
        input = {
            "prompt": prompt_list[idx%100],
            "height": 512,
            "width": 512,
            "inference_steps": steps_list[idx%100],
            "lora_tag": lora_list[idx%100],
            "guidance_scale": 7.0,
            "uuid": uuid.uuid1(),
            "request_time": time.time(),
            "SLOs": 4 * profile_latency[str(512)][str(steps_list[idx%100])],
            #"SLOs": 4 * profile_latency[str(size_list[idx%100])][str(steps_list[idx%100])],
            "id": idx # for debug
        }
        clip_queue.put(input)
        print(f"clip_queue put item: {input}""\n""-----------------------")
        # clip_queue.put(get_one_workload())
        # Feeding data to the first queue
        #for i in range(5):  # just as an example
        #    clip_queue.put(f"data_{i}")
    #clip_queue.put(None)  # Signal the end of input for clip_worker

    for p in processes:
        p.join()
        #break
    print("throughput:", loop_count / (time.time() - begin))
