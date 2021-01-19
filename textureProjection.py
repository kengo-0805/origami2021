import os

import cv2
from PIL import Image
import pyglet
import pyglet.gl as gl
import pyrealsense2 as rs
from pyglet import clock

import math
import numpy as np
import time

import cornerDetect as ct

start = time.time()
# ===============================
# 定数
# ===============================
DATA_DIRNAME = "data"
DATA_DIRPATH = os.path.join(os.path.dirname(__file__), DATA_DIRNAME)
if not os.path.exists(DATA_DIRPATH):
    os.makedirs(DATA_DIRPATH)

CHESS_HNUM = 7  # 水平方向個数
CHESS_VNUM = 10  # 垂直方向個数
CHESS_MARGIN = 50  # [px]
CHESS_BLOCKSIZE = 80  # [px]

BOARD_WIDTH = 0.02  # chessboard の横幅 [m]
BOARD_HEIGHT = 0.01  # chessboard の縦幅 [m]
BOARD_X = 0.  # chessboard の3次元位置X座標 [m]（右手系）
BOARD_Y = 0.  # chessboard の3次元位置Y座標 [m]（右手系）
BOARD_Z = 0.41  # chessboard の3次元位置Z座標 [m]（右手系）

dt_anim = 17
num = 0
# tex_X0 = 74
# tex_X1 = 12
# tex_X2 = 79
# tex_X3 = 16
# tex_Y0 = 18
# tex_Y1 = 23
# tex_Y2 = 81
# tex_Y3 = 85
tex_X = []
tex_Y = []

# OpenGL の射影のパラメータ
class Params:
    def __init__(self, zNear=0.0001, zFar=20.0, fovy=21.0):
        self.Z_NEAR = zNear  # 最も近い点 [m]
        self.Z_FAR = zFar  # 最も遠い点 [m]
        self.FOVY = fovy  # 縦の視野角 [deg]


PARAMS = Params(zNear=0.0001,  # [m]
                zFar=20.0,  # [m]
                fovy=21.0  # [deg]
                )

'''
# ストリーム(Color/Depth)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)
'''

# ===============================
# グローバル変数
# ===============================
window = None  # pyglet の Window　クラスのインスタンス
state = None  # アプリの状態を管理する変数（AppState）
cam_w, cam_h = 0, 0  # 画面解像度


class AppState:
    def __init__(self, params):
        self.params = params
        self.pitch = math.radians(0)
        self.yaw = math.radians(0)
        self.translation = np.array([0, 0, 0], np.float32)
        self.distance = 0
        self.lighting = False
        self.zNear = self.params.Z_NEAR


# テクスチャの自作
def make_chessboard(num_h, num_v, margin, block_size):
    chessboard = np.ones((block_size * num_v + margin * 2, block_size * num_h + margin * 2, 3), dtype=np.uint8) * 255

    for y in range(num_v):
        for x in range(num_h):
            if (x + y) % 2 == 0:
                sx = x * block_size + margin
                sy = y * block_size + margin
                chessboard[sy:sy + block_size, sx:sx + block_size, 0] = 0
                chessboard[sy:sy + block_size, sx:sx + block_size, 1] = 0
                chessboard[sy:sy + block_size, sx:sx + block_size, 2] = 0
    return chessboard


# 描画する画像の用意
def load_chessboard():
    global texture_ids, chessboard_image

    # chessboard = make_chessboard(CHESS_HNUM, CHESS_VNUM, CHESS_MARGIN, CHESS_BLOCKSIZE)

    # filepath = os.path.join(DATA_DIRPATH, 'chessboard.png')
    filepath = ("fig/dotdotdot.png")

    # cv2.imwrite(filepath, chessboard)
    chessboard_image = Image.open(filepath)

    tw, th = chessboard_image.width, chessboard_image.height
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_ids[0])
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, tw, th, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                    chessboard_image.tobytes())


#   座標の指定と描画
def board():
    global chessboard_image, texture_ids
    # alfa = 0.035
    alfa = 0

    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_ids[0])
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)

    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glTranslatef(0.5 / chessboard_image.width, 0.5 / chessboard_image.height, 0)
    # テクスチャの頂点の指定
    gl.glBegin(gl.GL_QUADS)

    # # 座標たち
    # gl.glTexCoord2i(0, 0)  # テクスチャ座標（左上）id = 0
    # gl.glVertex3f(-0.145, -0.16, BOARD_Z)
    # gl.glTexCoord2i(0, 1)  # テクスチャ座標（左下）id = 3
    # gl.glVertex3f(-0.145, BOARD_Y, BOARD_Z)
    # gl.glTexCoord2i(1, 1)  # テクスチャ座標（右下）id = 2
    # gl.glVertex3f(0.145, BOARD_Y, BOARD_Z)
    # gl.glTexCoord2i(1, 0)  # テクスチャ座標（右上）id = 1
    # gl.glVertex3f(0.145, -0.16, BOARD_Z)
    # gl.glEnd()
    # gl.glPopMatrix()


# # 座標たち
#     gl.glTexCoord2i(0, 0) # テクスチャ座標（左上）id = 0
#     gl.glVertex3f(-0.0670625, -0.136888889, BOARD_Z)
#     gl.glTexCoord2i(0, 1) # テクスチャ座標（左下）id = 3
#     gl.glVertex3f(-0.0525625, -0.023111111, BOARD_Z)
#     gl.glTexCoord2i(1, 1) # テクスチャ座標（右下）id = 2
#     gl.glVertex3f(0.0598125, -0.035555556, BOARD_Z)
#     gl.glTexCoord2i(1, 0) # テクスチャ座標（右上）id = 1
#     gl.glVertex3f(0.0489375, -0.147555556, BOARD_Z)
#     gl.glEnd()
#     gl.glPopMatrix()


    # global tex_X0 
    # global tex_X1 
    # global tex_X2 
    # global tex_X3
    # global tex_Y0 
    # global tex_Y1
    # global tex_Y2
    # global tex_Y3
#     print("左上")
#     print((tex_X[1]*0.29/160) - 0.145, -(tex_Y[1]*0.16/90))
#     print("左下")
#     print((tex_X[3]*0.29/160) - 0.145, -(tex_Y[3]*0.16/90))
#     print("右下")
#     print((tex_X[2]*0.29/160) - 0.145, -(tex_Y[2]*0.16/90))
#     print("右上")
#     print((tex_X[0]*0.29/160) - 0.145, -(tex_Y[0]*0.16/90))
# 座標たち
    gl.glTexCoord2i(0, 0) # テクスチャ座標（左上）id = 0
    gl.glVertex3f((tex_X[1]*0.29/160) - 0.145 - alfa, -(tex_Y[1]*0.16/90) + alfa, BOARD_Z)
    gl.glTexCoord2i(0, 1) # テクスチャ座標（左下）id = 3
    gl.glVertex3f((tex_X[3]*0.29/160) - 0.145 - alfa, -(tex_Y[3]*0.16/90) + alfa, BOARD_Z)
    gl.glTexCoord2i(1, 1) # テクスチャ座標（右下）id = 2
    gl.glVertex3f((tex_X[2]*0.29/160) - 0.145 - alfa, -(tex_Y[2]*0.16/90) + alfa, BOARD_Z)
    gl.glTexCoord2i(1, 0) # テクスチャ座標（右上）id = 1
    gl.glVertex3f((tex_X[0]*0.29/160) - 0.145 - alfa, -(tex_Y[0]*0.16/90) + alfa, BOARD_Z)
    gl.glEnd()
    gl.glPopMatrix()

def run_animation(dt):
    global num, tex_X, tex_Y
    # num = 0
    print(num)
    ret, frame = cap.read()
    if ret == True:
        cv2.imwrite("pic/picture{:0=3}".format(num) + ".png", frame)
        # print("save picture{:0=3}".format(num)+".png")
        ct.changePosition(num)
        ct.resize()
        x, y = ct.cornerDetect()
        num += 1
        # print("渡した座標")
        # print(x, y)
        tex_X = x
        tex_Y = y
        # print(tex_X,tex_Y)
    # global tex_X0 
    # global tex_X1 
    # global tex_X2 
    # global tex_X3 
    # global tex_Y0 
    # global tex_Y1 
    # global tex_Y2 
    # global tex_Y3 

    # tex_X0 += 1
    # tex_X1 += 1
    # tex_X2 += 1
    # tex_X3 += 1
    # tex_Y0 += 1
    # tex_Y1 += 1
    # tex_Y2 += 1
    # tex_Y3 += 1
    if len(x) == 4:
        print("投影")
    # board()
        on_draw_impl()

# 描画の世界を作っている
def on_draw_impl():
    window.clear()

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LINE_SMOOTH)

    width, height = window.get_size()
    # print(width,height)
    # width = 2560
    # height = 1600
    gl.glViewport(0, 0, width, height)

    # 射影行列の設定
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    aspect = width / float(height * 2)
    bottom = 0
    top = state.zNear * np.tan(np.radians(PARAMS.FOVY))
    left = - top * aspect
    right = top * aspect
    gl.glFrustum(left, right, bottom, top, state.zNear, PARAMS.Z_FAR)  # 視野錐台の大事なやつ

    # 視点の設定
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.gluLookAt(0.0, 0.0, -0.1,
                 0.0, 0.0, 1.0,
                 0.0, -1.0, 0.0)

    gl.glTranslatef(0, 0, state.distance)
    gl.glRotated(state.pitch, 1, 0, 0)
    gl.glRotated(state.yaw, 0, 1, 0)

    gl.glTranslatef(0, 0, -state.distance)
    gl.glTranslatef(*state.translation)
    # * は分解して渡すことを意味している
    # gl.glTranslatef(*[a,b,c]) は gl.glTranslatef(a,b,c) と同じ

    if state.lighting:
        ldir = [0.5, 0.5, 0.5]  # world-space lighting
        ldir = np.dot(state.rotation, (0, 0, 1))  # MeshLab style lighting
        ldir = list(ldir) + [0]  # w=0, directional light
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*ldir))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,
                     (gl.GLfloat * 3)(1.0, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,
                     (gl.GLfloat * 3)(0.75, 0.75, 0.75))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_LIGHTING)

    # comment this to get round points with MSAA on
    gl.glEnable(gl.GL_POINT_SPRITE)

    board()
    # line()

# -------------------------------
# ここからがメイン部分
# -------------------------------

texture_ids = (pyglet.gl.GLuint * 1)()
gl.glGenTextures(1, texture_ids)
load_chessboard()

# アプリクラスのインスタンス
state = AppState(PARAMS)
platform = pyglet.window.get_platform()
display = platform.get_default_display()
screens = display.get_screens()
window = pyglet.window.Window(
    config=gl.Config(
        double_buffer=True,
        samples=8  # MSAA
    ),
    resizable=True,
    vsync=True,
    fullscreen=True,
    screen=screens[0])


@window.event
def on_draw():
    pyglet.clock.schedule_interval(run_animation, dt_anim / 1000.0)
    # on_draw_impl()

video_path = 1
cap = cv2.VideoCapture(video_path)
# pyglet.clock.schedule_interval(run_animation, dt_anim / 1000.0)
# run_animation(dt)
pyglet.app.run()

# elapsed_time = time.time() - start
# # if @window.event = 1
# print("elapsed_time:{0}".format(elapsed_time) + "[sec]")