import cv2
import numpy as np
import mediapipe as mp
import json
import time
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

app = FastAPI()

# --- 1. SETUP AI MODELS ---
print("Loading Grocery POS Models...")

# Load Object Detection (Using .pt format for now)
try:
    model = YOLO('best_v12.pt')
    print("✅ Custom Torch Model Loaded (best.pt)")
except Exception as e:
    print(f"⚠️ Custom model not found ({e}). Using Standard YOLO.")
    model = YOLO('yolo11n.pt')

# Load Hand Gesture Recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. PRODUCT DATABASE ---
# Mapped from data.yaml classes (34 products)
ITEM_DB = {
    0: {"name": "Nescafe Coffee", "price": 12.00},
    1: {"name": "Kopiko Coffee", "price": 15.00},
    2: {"name": "Lucky Me Pancit Canton", "price": 25.00},
    3: {"name": "Coke in Can", "price": 45.00},
    4: {"name": "Alaska Milk", "price": 55.00},
    5: {"name": "Century Tuna", "price": 42.00},
    6: {"name": "VCut Spicy BBQ", "price": 38.00},
    7: {"name": "Selecta Cornetto", "price": 30.00},
    8: {"name": "Nestle Yogurt", "price": 35.00},
    9: {"name": "Femme Tissue", "price": 20.00},
    10: {"name": "Maya Champorado", "price": 40.00},
    11: {"name": "J&J Potato Chips", "price": 35.00},
    12: {"name": "Nivea Deodorant", "price": 89.00},
    13: {"name": "UFC Canned Mushroom", "price": 32.00},
    14: {"name": "Libby's Sausage", "price": 50.00},
    15: {"name": "Stik-O", "price": 65.00},
    16: {"name": "Nissin Cup Noodles", "price": 28.00},
    17: {"name": "Dewberry Strawberry", "price": 75.00},
    18: {"name": "Smart-C", "price": 35.00},
    19: {"name": "Pineapple Juice", "price": 40.00},
    20: {"name": "Nestle Chuckie", "price": 32.00},
    21: {"name": "Delight Probiotic", "price": 10.00},
    22: {"name": "Summit Water", "price": 20.00},
    23: {"name": "Almond Milk", "price": 120.00},
    24: {"name": "Piknik", "price": 85.00},
    25: {"name": "Bactidol", "price": 150.00},
    26: {"name": "Head & Shoulders", "price": 12.00},
    27: {"name": "Irish Spring Soap", "price": 45.00},
    28: {"name": "C2 Green Tea", "price": 28.00},
    29: {"name": "Colgate Toothpaste", "price": 95.00},
    30: {"name": "555 Sardines", "price": 22.00},
    31: {"name": "Meadows Truffle Chips", "price": 140.00},
    32: {"name": "Double Black", "price": 60.00},
    33: {"name": "Nongshim Noodles", "price": 55.00},
}

# --- 3. HELPER FUNCTIONS ---
def get_gesture_state(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]   
    extended_count = 0
    for i in range(4):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y:
            extended_count += 1
          
    if extended_count >= 4: return "OPEN"
    elif extended_count == 0: return "CLOSED"
    else: return "UNKNOWN"

# --- 4. STATE MANAGEMENT ---
class KioskState:
    def __init__(self):
        self.mode = "IDLE"
        self.cart = []            
        self.total = 0.0
        self.last_scan_time = 0
        self.cooldown = 2.5
        self.last_gesture = "UNKNOWN"
        self.gesture_debounce = 0
        self.gesture_timeout = 3.0  # Seconds to complete OPEN->CLOSED sequence
        self.last_gesture_time = 0

state = KioskState()

# --- 5. WEBSOCKET LOOP ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("POS Client Connected")
    
    try:
        while True:
            # A. Receive Image
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # B. Prepare Response
            response = {
                "mode": state.mode,
                "feedback": "",
                "cart": state.cart,
                "total": state.total,
                "boxes": [],            
                "hand_coords": [],      
                "gesture": "UNKNOWN"
            }
            
            # --- C. OBJECT DETECTION (Run First!) ---
            product_detected = False
            
            # Check for items in SCANNING mode (Safety Lock)
            if state.mode == "SCANNING":
                 results = model(frame, conf=0.6, verbose=False)
                 for r in results:
                    if len(r.boxes) > 0:
                        product_detected = True
                        
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        label_name = model.names[int(box.cls[0])]
                        
                        response["boxes"].append({
                            "coords": [x1, y1, x2, y2],
                            "label": label_name
                        })
                        
                        # Add to Cart Logic
                        cls_id = int(box.cls[0])
                        if cls_id in ITEM_DB:
                            item = ITEM_DB[cls_id]
                            scan_time = time.time()
                            if scan_time - state.last_scan_time > state.cooldown:
                                # Check if item already in cart
                                found = False
                                for cart_item in state.cart:
                                    if cart_item["name"] == item["name"]:
                                        cart_item["quantity"] += 1
                                        found = True
                                        break
                                
                                if not found:
                                    state.cart.append({**item, "quantity": 1})
                                
                                state.total += item["price"]
                                state.last_scan_time = scan_time
                                response["feedback"] = f"Added {item['name']}!"
                                
                                # Play beep sound (non-blocking)
                                try:
                                    subprocess.Popen(['paplay', 'beep.wav'],
                                                   stdout=subprocess.DEVNULL, 
                                                   stderr=subprocess.DEVNULL)
                                except Exception as e:
                                    print(f"Sound error: {e}")  # Debug: see if sound fails

            # --- D. HAND GESTURE DETECTION ---
            current_gesture = "UNKNOWN"

            # CRITICAL CHECK: Only run gesture logic if NO product is detected
            if not product_detected:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(frame_rgb)

                if hand_results.multi_hand_landmarks:
                    hand_lms = hand_results.multi_hand_landmarks[0]
                    # Save coords for drawing
                    for lm in hand_lms.landmark:
                        response["hand_coords"].append([lm.x, lm.y])
                    
                    current_gesture = get_gesture_state(hand_lms)
                    response["gesture"] = current_gesture

            # --- E. GESTURE TRANSITION LOGIC ---
            trigger_action = False
            curr_time = time.time()

            if current_gesture != "UNKNOWN":
                # Check if last gesture has expired
                if curr_time - state.last_gesture_time > state.gesture_timeout:
                    state.last_gesture = "UNKNOWN"
                
                # Detect OPEN->CLOSED sequence within timeout
                if curr_time - state.gesture_debounce > 1.5:
                    if state.last_gesture == "OPEN" and current_gesture == "CLOSED":
                        # Verify OPEN happened recently (within timeout)
                        if curr_time - state.last_gesture_time <= state.gesture_timeout:
                            trigger_action = True
                            state.gesture_debounce = curr_time
                            print("👉 OPEN→CLOSED DETECTED!")
                
                # Update gesture state
                if current_gesture != state.last_gesture:
                    state.last_gesture = current_gesture
                    state.last_gesture_time = curr_time
            else:
                state.last_gesture = "UNKNOWN"

            # --- F. STATE MACHINE ---
            if state.mode == "IDLE":
                response["feedback"] = "Open Hand ✋ then Fist ✊ to Start"
                if trigger_action:
                    state.mode = "SCANNING"
                    state.cart = []
                    state.total = 0.0
                    print("Transition: IDLE -> SCANNING")

            elif state.mode == "SCANNING":
                if response["feedback"] == "":
                    response["feedback"] = "Scanning... Open Hand ✋ then Fist ✊ to Pay"
                
                if trigger_action:
                    state.mode = "PAID"
                    state.cart = []
                    state.total = 0.0
                    print("Transition: SCANNING -> PAID")

            elif state.mode == "PAID":
                response["feedback"] = "Paid! Resetting..."
                if time.time() - state.gesture_debounce > 2.0:
                     state.mode = "IDLE"

            # G. Send JSON
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Client Disconnected")
        state.mode = "IDLE"
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
