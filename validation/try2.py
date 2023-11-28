
import pandas as pd
import os
import shutil

path1 = os.path.abspath(__file__)
print(path1)
#df = pd.read_csv("/content/drive/MyDrive/project182/emoji_image/full_emoji.csv",encoding='latin-1')
#print(df[['name']][:100])
#
#df_emoji = df[['name']][:700]
#df_model = df[['name']][:100]
#
##for i in range(100):
#  # df_Apple.loc[i, 'path'] = "/content/drive/MyDrive/project182/emoji_image/emoji/" + "Apple" + str(i) + ".jpg"
#  #df_Apple.loc[i, 'name'] = df_Apple.loc[i, 'name'] + " in Apple style"
#  #df_Apple.loc[i, 'image_path'] = 'Apple' + str(i + 1) + ".png"
##print(df_Apple)
#i = 0
#j = 0
#while i != 7:
#  j = 0
#  while j != 100:
#    base = i * 100
#    if j + 1 >= 63:
#      cur = j + 2
#    elif j + 1 <= 21:
#      cur = j
#    else:
#      cur = j + 1
#    if i == 0:
#      df_emoji.loc[base + j, 'name'] = df_model.loc[j, 'name'] + " in Apple style"
#      df_emoji.loc[base + j, 'image_path'] = 'Apple' + str(cur + 1) + ".png"
#    elif i == 1:
#      df_emoji.loc[base + j, 'name'] = df_model.loc[j, 'name'] + " in Facebook style"
#      df_emoji.loc[base + j, 'image_path'] = 'Facebook' + str(cur + 1) + ".png"
#    elif i == 2:
#      df_emoji.loc[base + j, 'name'] = df_model.loc[j, 'name'] + " in Google style"
#      df_emoji.loc[base + j, 'image_path'] = 'Google' + str(cur + 1) + ".png"
#    elif i == 3:
#      df_emoji.loc[base + j, 'name'] = df_model.loc[j, 'name'] + " in JoyPixels style"
#      df_emoji.loc[base + j, 'image_path'] = 'JoyPixels' + str(cur + 1) + ".png"
#    elif i == 4:
#      df_emoji.loc[base + j, 'name'] = df_model.loc[j, 'name'] + " in Samsung style"
#      df_emoji.loc[base + j, 'image_path'] = 'Samsung' + str(cur + 1) + ".png"
#    elif i == 5:
#      df_emoji.loc[base + j, 'name'] = df_model.loc[j, 'name'] + " in Twitter style"
#      df_emoji.loc[base + j, 'image_path'] = 'Twitter' + str(cur + 1) + ".png"
#    elif i == 6:
#      df_emoji.loc[base + j, 'name'] = df_model.loc[j, 'name'] + " in Windows style"
#      df_emoji.loc[base + j, 'image_path'] = 'Windows' + str(cur + 1) + ".png"
#    j = j + 1
#  i = i + 1
#print(df_emoji)
#
#df_emoji.to_csv("/content/drive/MyDrive/project182/emoji_image/emoji.csv", index=False)