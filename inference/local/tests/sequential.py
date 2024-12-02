from handler_test import MyHandler

handler = MyHandler()

data1 = {"instances": [{"data": [13,13,13,0,13,13,13,13,-76,-76,-81,-76,-78.5,-76,-76,-7,-7,-12,-7,-9.5,-7,-7,12,12,7,12,9.5,12,12]}]}
data2 = {"instances": [{"data": [13,13,12,0.4714045208,13,12.5,13,13,-81,-81,-82,-81,-81.5,-81,-81,-12,-12,-11,-12,-12,-12,-11.5,7,7,2,7,4.5,7,7]}]}
data3 = {"instances": [{"data": [6,7,7,0.4714045208,7,6.5,7,7,-107,-106,-106,-106,-106.5,-106,-106,-13,-14,-14,-14,-14,-14,-13.5,2,2,2,2,2,2,2]}]}
data4 = {"instances": [{"data": [8,8,8,0,8,8,8,8,-108,-108,-108,-108,-108,-108,-108,-13,-13,-13,-13,-13,-13,-13,2,2,2,2,2,2,2]}]}

result1 = handler.handle(data1)
print("result 1:", result1)

result2 = handler.handle(data2)
print("result 2:", result2)

result3 = handler.handle(data3)
print("result 3:", result3)

result4 = handler.handle(data4)
print("result 4:", result4)