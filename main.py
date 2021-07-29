import cv2
import time
import math
import autopy
import numpy as np
import mediapipe as mp
import sys

class handDetector():
	def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
		self.mode = mode
		self.maxHands = maxHands
		self.detectionConf = detectionConf
		self.trackConf = trackConf
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackConf)
		self.mpDraw = mp.solutions.drawing_utils

	def findHands(self, img):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(imgRGB)

		if self.results.multi_hand_landmarks:
			for hand_landmarks in self.results.multi_hand_landmarks:
				self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
		return img

	def findPosition(self, img, no = 0, Draw = True):
		lmList = []

		if self.results.multi_hand_landmarks:
			targetHand = self.results.multi_hand_landmarks[no]
			for idx, lm in enumerate(targetHand.landmark):
				h, w, c = img.shape
				cx = int(lm.x * w)
				cy = int(lm.y * h)
				lmList.append([idx, cx, cy])
				if Draw:
					cv2.circle(img, (cx, cy), 7, (255,255,0), cv2.FILLED)
		return lmList

	def fingerUp(self, lmList):
		uplist = [0,0,0,0,0]

		if abs(lmList[4][1]-lmList[9][1]) > abs(lmList[2][1]-lmList[9][1]): uplist[0] = 1
		if abs(lmList[8][2]-lmList[0][2]) > abs(lmList[6][2]-lmList[0][2]): uplist[1] = 1
		if abs(lmList[12][2]-lmList[0][2]) > abs(lmList[10][2]-lmList[0][2]): uplist[2] = 1
		if abs(lmList[16][2]-lmList[0][2]) > abs(lmList[14][2]-lmList[0][2]): uplist[3] = 1
		if abs(lmList[20][2]-lmList[0][2]) > abs(lmList[18][2]-lmList[0][2]): uplist[4] = 1 

		return uplist

	def calcDistance(self, p1, p2):
		return math.sqrt(((p1[1]-p2[1])**2) + ((p1[2]-p2[2])**2))

def Draw():
	last_x, last_y = 0,0
	cap = cv2.VideoCapture(0)
	canvas = np.zeros((480,640,3), np.uint8)

	detector = handDetector()

	now_color = (0,0,0)

	while cap.isOpened():
		success, img = cap.read()
		img = cv2.flip(img, 1)
		if not success:
			print('not success')
			continue
		img = detector.findHands(img)
		lmList = detector.findPosition(img, Draw = False)

		cv2.rectangle(img, (0,0), (160,60), (255,0,0), -1)
		cv2.rectangle(img, (160,0), (320,60), (0,255,0), -1)
		cv2.rectangle(img, (320,0), (480,60), (0,0,255), -1)
		cv2.rectangle(img, (480,0), (640,60), (0,0,0), -1)


		if len(lmList) != 0:
			x = lmList[8][1]
			y = lmList[8][2]
			up = detector.fingerUp(lmList)

			mode = [up[1],up[2],up[3]]

			if mode == [1,0,0]:
				cv2.putText(img, 'drawing mode', (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)			
				if last_x == 0 and last_y == 0:
					last_x, last_y = x, y
				if now_color == (0,0,0):
					last_x, last_y = 0, 0
					cv2.circle(canvas, (x,y), 15, (0,0,0), cv2.FILLED)
				else:
					cv2.line(canvas, (last_x,last_y), (x,y), now_color, 5)
					last_x, last_y = x,y
			elif mode == [1,1,0]:
				cv2.putText(img, 'selecting mode', (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
				last_x, last_y = 0,0
				if 0 <= x < 160 and 0 <= y <= 60: now_color = (255,0,0) 
				elif 160 <= x < 320 and 0 <= y <= 60: now_color = (0,255,0)
				elif 320 <= x < 480 and 0 <= y <= 60: now_color = (0,0,255)
				elif 480 <= x and 0 <= y <= 60: now_color = (0,0,0)
			cv2.circle(img, (x,y), 15, now_color, cv2.FILLED)
			# elif mode == [1,1,1]:
			# 	cv2.putText(img, 'erasing mode', (10,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
			# 	last_x, last_y = 0,0
			# 	cv2.circle(img, (x,y), 15, (255,0,0), cv2.FILLED)
			# 	cv2.circle(canvas, (x,y), 15, (0,0,0), cv2.FILLED)

		canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
		_, canvasInv =  cv2.threshold(canvasGray, 50, 255, cv2.THRESH_BINARY_INV)
		canvasInv = cv2.cvtColor(canvasInv, cv2.COLOR_GRAY2BGR)
		img = cv2.bitwise_and(img, canvasInv)
		img = cv2.bitwise_or(img, canvas)

		# added_img = cv2.addWeighted(img,0.4,canvas,0.1,0)
		cv2.imshow('image', img)
		cv2.imshow('canvas', canvas)
		if cv2.waitKey(5) & 0xFF == 27:
			break
	cap.release()

def Mouse():
	last_x, last_y = 0,0
	screen_w, screen_h = autopy.screen.size()
	# print(screen_w, screen_h)
	lastTime = 0
	nowTime = 0
	cap = cv2.VideoCapture(0)
	cam_w = cap.get(3)
	cam_h = cap.get(4)
	detector = handDetector()
	while cap.isOpened():
		success, img = cap.read()
		img = cv2.flip(img, 1)
		if not success:
			print('not success')
			continue
		img = detector.findHands(img)
		lmList = detector.findPosition(img, Draw=False)
		# print(lmList)
		cv2.rectangle(img, (200,0), (int(cam_w), int(cam_h-190)), (255,255,255), 2)

		if len(lmList) != 0:
			# 	print(lmList[8])
			x1 = lmList[8][1]
			y1 = lmList[8][2]
			string = '(' + str(x1) + ',' + str(y1) + ')'
			# cv2.putText(img, string, (10,140), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
			up = detector.fingerUp(lmList)
			# string = str(up.count(1)) + ' finger(s) up'
			# cv2.putText(img, string, (10,140), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

			dist = detector.calcDistance(lmList[3], lmList[5])

			if x1 < 500 or y1 < 300:
				if up[1] and not up[0]:
					print('moving mode')
					new_x = np.interp(x1, (200,cam_w), (0,screen_w-0.1))
					new_y = np.interp(y1, (0,cam_h-190), (0,screen_h-0.1))
					
					sm_x = last_x + (new_x - last_x) / 7
					sm_y = last_y + (new_y - last_y) / 7

					# print(new_x, new_y)
					autopy.mouse.move(sm_x, sm_y)

					last_x, last_y = sm_x, sm_y

				elif up[1] and up[0]:
					# print('clicking mode')
					if dist < 40:
						print('CLICK')
						autopy.mouse.click()

				cv2.line(img, (lmList[3][1],lmList[3][2]), (lmList[5][1],lmList[5][2]), [255, 255, 0], 2)
				cv2.putText(img, str(dist), (10,140), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

		nowTime = time.time()
		fps = 1 / (nowTime - lastTime)
		lastTime = nowTime

		cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
		cv2.imshow('image', img)
		if cv2.waitKey(5) & 0xFF == 27:
			break
	cap.release()

def main():
	print(sys.argv[1])
	if sys.argv[1] == 'mouse':
		Mouse()
	elif sys.argv[1] == 'draw':
		Draw()

if __name__ == '__main__':
	main()