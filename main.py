import torch
import data_processor as dpr

if __name__ == '__main__':

    # 一个测试用的小脚本
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('data/model.pt')
    model.to(device)

    # 推理模式
    model.eval()

    # 测试数据
    sms = '【瑞幸咖啡】请查收→超全薅“鹿”毛攻略！内含4.5折惊喜券，戳→ yyds.co 解锁 谨防泄漏取阅回N'
    sms_vector = dpr.get_sequence(sms)

    # 推理
    with torch.no_grad():
        output = model(sms_vector.to(device))
    prediction = torch.round(torch.sigmoid(output)).item()
    if prediction == 1:
        print('短信内容为：', sms)
        print('这是一条垃圾短信')
    else:
        print('短信内容为：', sms)
        print('这是一条正常短信')
