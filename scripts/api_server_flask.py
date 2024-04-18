from flask import Flask, request, jsonify, abort
import rospy
from create_embeeding_grid_map.srv import ProcessText, ProcessTextRequest

app = Flask(__name__)

# Initialize ROS node
rospy.init_node('web_service_client', anonymous=True)

@app.route('/process_command/', methods=['POST'])
def process_command():
    # Parse the JSON request
    data = request.get_json()
    
    if not data or 'input' not in data:
        # If there's no input in the JSON data, return a 400 Bad Request error
        abort(400, description="Missing 'input' field in the request.")

    input_text = data['input']

    rospy.wait_for_service('/process_text')
    try:
        # Create service proxy
        process_text = rospy.ServiceProxy('/process_text', ProcessText)
        
        # Create request message
        req = ProcessTextRequest(input=input_text)
        
        # Call service and get response
        resp = process_text(req)
        
        # Assuming resp has an attribute 'output', adjust according to your response type
        return jsonify({"response": resp.output})
    except rospy.ServiceException as e:
        # Return a 500 Internal Server Error if something goes wrong with the ROS service
        abort(500, description=str(e))

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=18000, debug=True)

