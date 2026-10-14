import cv2
import mediapipe as mp
import numpy as np
import random
import math

# ================= CAMERA =================

camera = cv2.VideoCapture(0)

# ================= HAND DETECTION =================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= FLOWER DRAWING =================

def flower(img, x, y, size, flower_type):

    x = int(x)
    y = int(y)
    size = int(size)

    # ---------- DAISY ----------
    if flower_type == 0:

        for angle in range(0, 360, 45):

            a = math.radians(angle)

            px = int(x + math.cos(a) * size)
            py = int(y + math.sin(a) * size)

            cv2.ellipse(
                img,
                (px, py),
                (size // 2, size),
                angle,
                0,
                360,
                (255, 255, 255),
                -1
            )

        cv2.circle(
            img,
            (x, y),
            size // 2,
            (0, 220, 255),
            -1
        )

    # ---------- SUNFLOWER ----------
    elif flower_type == 1:

        for angle in range(0, 360, 30):

            a = math.radians(angle)

            px = int(x + math.cos(a) * size)
            py = int(y + math.sin(a) * size)

            cv2.ellipse(
                img,
                (px, py),
                (size // 2, size),
                angle,
                0,
                360,
                (0, 180, 255),
                -1
            )

        cv2.circle(
            img,
            (x, y),
            size // 2,
            (40, 60, 80),
            -1
        )

    # ---------- ROSE ----------
    elif flower_type == 2:

        cv2.circle(
            img,
            (x, y),
            size,
            (30, 30, 220),
            -1
        )

        for angle in range(0, 360, 60):

            a = math.radians(angle)

            px = int(x + math.cos(a) * size * 0.65)
            py = int(y + math.sin(a) * size * 0.65)

            cv2.circle(
                img,
                (px, py),
                size // 2,
                (70, 40, 255),
                -1
            )

        cv2.circle(
            img,
            (x, y),
            size // 3,
            (20, 20, 150),
            -1
        )

    # ---------- TULIP ----------
    elif flower_type == 3:

        pts = np.array([
            [x - size, y - size],
            [x - size // 2, y - size // 3],
            [x, y - size],
            [x + size // 2, y - size // 3],
            [x + size, y - size],
            [x + size // 2, y + size],
            [x - size // 2, y + size]
        ])

        cv2.fillPoly(
            img,
            [pts],
            (180, 80, 255)
        )

        cv2.line(
            img,
            (x, y + size),
            (x, y + size * 3),
            (30, 180, 40),
            4
        )

    # ---------- CHERRY BLOSSOM ----------
    else:

        for angle in range(0, 360, 72):

            a = math.radians(angle)

            px = int(x + math.cos(a) * size * 0.8)
            py = int(y + math.sin(a) * size * 0.8)

            cv2.circle(
                img,
                (px, py),
                size // 2,
                (180, 120, 255),
                -1
            )

        cv2.circle(
            img,
            (x, y),
            size // 3,
            (80, 220, 255),
            -1
        )


# ================= FLOATING FLOWERS =================

floating = []

for i in range(25):

    floating.append({
        "x": random.randint(30, 1200),
        "y": random.randint(30, 700),
        "size": random.randint(8, 18),
        "speed": random.uniform(0.3, 1.2),
        "type": random.randint(0, 4)
    })


# ================= DRAWING FLOWERS =================

trail = []

last_point = None


# ================= BURST =================

particles = []


def flower_burst(x, y):

    for i in range(45):

        angle = random.uniform(0, math.pi * 2)

        speed = random.uniform(3, 9)

        particles.append({
            "x": x,
            "y": y,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "life": 50,
            "size": random.randint(8, 18),
            "type": random.randint(0, 4)
        })


# ================= MAIN LOOP =================

while True:

    success, frame = camera.read()

    if not success:
        print("Camera not found")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)


    # ================= FLOATING FLOWERS =================

    for f in floating:

        f["y"] -= f["speed"]

        if f["y"] < -30:

            f["y"] = h + 30

            f["x"] = random.randint(
                20,
                w - 20
            )

        flower(
            frame,
            f["x"],
            f["y"],
            f["size"],
            f["type"]
        )


    # ================= HAND =================

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Finger landmarks

            index = hand.landmark[8]
            middle = hand.landmark[12]
            ring = hand.landmark[16]
            pinky = hand.landmark[20]

            index_pip = hand.landmark[6]
            middle_pip = hand.landmark[10]
            ring_pip = hand.landmark[14]
            pinky_pip = hand.landmark[18]


            # Screen position

            ix = int(index.x * w)
            iy = int(index.y * h)


            # Finger states

            index_up = index.y < index_pip.y
            middle_up = middle.y < middle_pip.y
            ring_up = ring.y < ring_pip.y
            pinky_up = pinky.y < pinky_pip.y


            # ================= OPEN HAND =================

            open_hand = (
                index_up and
                middle_up and
                ring_up and
                pinky_up
            )


            # ================= DRAWING =================

            if (
                index_up and
                not middle_up and
                not ring_up and
                not pinky_up
            ):

                if last_point is not None:

                    distance = math.sqrt(
                        (ix - last_point[0]) ** 2 +
                        (iy - last_point[1]) ** 2
                    )

                    if distance > 10:

                        trail.append({
                            "x": ix,
                            "y": iy,
                            "life": 100,
                            "type": random.randint(0, 4)
                        })

                last_point = (ix, iy)

            else:

                last_point = None


            # ================= OPEN HAND BURST =================

            if open_hand:

                flower_burst(
                    ix,
                    iy
                )


    # ================= DRAW TRAIL =================

    new_trail = []

    for p in trail:

        p["life"] -= 2

        if p["life"] > 0:

            flower(
                frame,
                p["x"],
                p["y"],
                10,
                p["type"]
            )

            new_trail.append(p)

    trail = new_trail


    # ================= BURST ANIMATION =================

    new_particles = []

    for p in particles:

        p["x"] += p["vx"]

        p["y"] += p["vy"]

        p["vy"] += 0.08

        p["life"] -= 1

        if p["life"] > 0:

            flower(
                frame,
                p["x"],
                p["y"],
                p["size"],
                p["type"]
            )

            new_particles.append(p)

    particles = new_particles


    # ================= TITLE =================

    cv2.putText(
        frame,
        "FLOWER MAGIC",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 100, 200),
        2
    )

    cv2.putText(
        frame,
        "INDEX FINGER = DRAW FLOWERS",
        (30, h - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 255),
        2
    )

    cv2.putText(
        frame,
        "OPEN PALM = FLOWER BURST",
        (30, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 255),
        2
    )


    # ================= SHOW =================

    cv2.imshow(
        "Flower Magic - Hand Gesture",
        frame
    )


    # ================= QUIT =================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()