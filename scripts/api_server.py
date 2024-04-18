from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rospy
from create_embeeding_grid_map.srv import ProcessText, ProcessTextRequest

app = FastAPI()

# 定义请求模型
class Command(BaseModel):
    input: str

# 初始化ROS节点
rospy.init_node('web_service_client', anonymous=True)

@app.post("/process_command/")
async def process_command(command: Command):
    rospy.wait_for_service('/process_text')
    try:
        # 创建服务代理
        process_text = rospy.ServiceProxy('/process_text', ProcessText)
        
        # 创建请求消息
        req = ProcessTextRequest(input=command.input)
        
        # 调用服务并获取响应
        resp = process_text(req)
        
        # 假设resp具有一个名为output的字段，你想返回这个字段
        # 请根据你的实际响应类型调整字段名
        return {"response": resp.output}
    except rospy.ServiceException as e:
        raise HTTPException(status_code=500, detail=str(e))
