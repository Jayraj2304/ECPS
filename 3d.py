from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import pandas as pd
df=pd.read_csv('with pos.csv')
pos=df["LOCATION"]
data_list = pos.values
print(data_list)

app = Ursina()
# Create a cube
player=FirstPersonController()
for p in data_list:
    n=p[1]

c=0
for block in range(int(p[1])):
    blocks1=[]

    for i in range (10):
        for j in range(20):
            block=Button(
                color=color.white,
                model='cube',
                position=(j,0,i+c),
                texture='Brick',
                parent=scene,
                    origin_y=0.5)
            blocks1.append(block)
    c=c+10

block1=data_list
for j in range(len(block1)):
    pos=block1[j]
    c=(int(pos[1])-1)*11
    if pos[2] == "A":
        x=2+c
    elif pos[2] == "B":
        x=4+c
    elif pos[2] == "C":
        x=6+c
    y=int(pos[3])
    z=int(pos[4])
    print(x,y,z)
    
    if z==2:
        block=Button(color=color.red,model='cube',position=(y+5,z,x),texture="Brick",parent=scene,origin_y=0.5)       
    else:
        block=Button(color=color.blue,model='cube',position=(y+5,z,x),texture="Brick",parent=scene,origin_y=0.5)  
    blocks1.append(block)   
app.run()   