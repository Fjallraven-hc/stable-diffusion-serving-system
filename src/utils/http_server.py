from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/square', methods=['POST'])
def square():
    try:
        data = request.json
        print(data)
        print(type(data))
        number = data['number']
        result = number ** 2
        return jsonify({'result': result})
    except KeyError:
        return jsonify({'error': 'Number is missing in the request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
