import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.type_check import imag
from PIL import Image

def changePosition(numnum):
  aruco = cv2.aruco
  p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
  img = cv2.imread("pic/picture{:0=3}".format(numnum)+".png")
  # img = cv2.imread("fig/square_risize.png")
  # print("画像を読み込んだ")
  corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict) # 検出

  # 時計回りで左上から順にマーカーの「中心座標」を m に格納
  m = np.empty((4,2)) # 空の行列を作る
  for i,c in zip(ids.ravel(), corners):
    m[i] = c[0].mean(axis=0)

  corners2 = [np.empty((1,4,2))]*4
  for i,c in zip(ids.ravel(), corners):
    corners2[i] = c.copy()
  m[0] = corners2[0][0][2]
  m[1] = corners2[1][0][3]
  m[2] = corners2[2][0][0]
  m[3] = corners2[3][0][1]

  width, height = (500,500) # 変形後画像サイズ
  marker_coordinates = np.float32(m)
  true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
  trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
  img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
  img_trans = cv2.cvtColor(img_trans, cv2.COLOR_BGR2RGB)
  # print(m[0],m[1],m[2],m[3])
  cv2.imwrite("fig/img_trans.png",img_trans)
  # plt.imshow(img_trans)
  # plt.show()

def resize():
  # 画像ファイル名指定
  file_name = "fig/img_trans.png"

  # 入力画像の読み込み
  img = Image.open(file_name)

  width, height = img.size

  # 画像の幅を表示
  # print('Original Width:', width)

  # 画像の高さを表示
  # print('Original Height:', height)

  # 任意のサイズに指定
  img_resize = img.resize((160, 90))
  width, height = img_resize.size
  # print('Resized Width:', width)
  # print('Resized Hight:', height)
  img_resize.save('fig/image_risize.png')

def cornerDetect():
  # 画像の読み込み
  image = cv2.imread("fig/image_risize.png")
  # グレイスケール
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  # エッジ処理
  edge = cv2.Canny(gray,400,20)
  # エッジ処理後の画像を書き込む
  cv2.imwrite("fig/edge_resize.png", edge)
  # 32bit化？
  edge = np.float32(edge)
  # コーナー検出
  dst = cv2.cornerHarris(edge,2,3,0.16)
  # 膨張処理
  dst = cv2.dilate(dst,None)
  # 赤い点をつける
  image[dst>0.01*dst.max()] = [255,0,0]
  # 赤い点の検知
  coord = np.where(np.all(image == (255, 0, 0), axis=-1))
  # 座標の表示
  corner_X = []
  corner_Y = []
  for i in range(len(coord[0])):
      # print("X:%s Y:%s"%(coord[1][i],coord[0][i]))
      original = coord[1][i] + coord[0][i]
      old = coord[1][i-1] + coord [0][i-1]
      if abs(original - old) > 15:  # XYを足した値が前の値から15以上変化していたら
        corner_X.append(coord[1][i])
        corner_Y.append(coord[0][i])
  for j in range(len(corner_X)):
    print("角X:%s 角Y:%s"%(corner_X[j],corner_Y[j]))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # 保存
  cv2.imwrite("fig/image_corner_resize.png",image)
  return corner_X, corner_Y


  # plt.imshow(image)
  # plt.title('cornerHarris image')
  # plt.show()




# video_path = 1
# cap = cv2.VideoCapture(video_path)
#
# # cap = VideoCapture()
#
# num = 1
# global frame
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         cv2.imwrite("pic/picture{:0=3}".format(num)+".png",frame)
#         # print("save picture{:0=3}".format(num)+".png")
#         changePosition()
#         resize()
#         cornerDetect()
#         # print(x, y)
#         num += 1
#     else:
#         break
#     time.sleep(3)
#
# cap.release()




# cornerDetect()