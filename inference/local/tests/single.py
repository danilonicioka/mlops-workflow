from handler import MyHandler

handler = MyHandler()

data = [{"data": [13,13,13,0,13,13,13,13,-76,-76,-81,-76,-78.5,-76,-76,-7,-7,-12,-7,-9.5,-7,-7,12,12,7,12,9.5,12,12]}]

result = handler.handle(data)

print("result: " ,result)