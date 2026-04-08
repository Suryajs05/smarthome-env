from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

app = FastAPI()

# --- Typed Models ---
class DeviceState(BaseModel):
    id: str
    type: str
    status: str
    temperature: Optional[float] = None

class Observation(BaseModel):
    time_of_day: str
    devices: List[DeviceState]
    total_power_watts: float
    feedback: str

class Action(BaseModel):
    device_id: str
    command: str # "turn_on", "turn_off", "set_temp"
    value: Optional[float] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    error: Optional[str] = None

class ResetRequest(BaseModel):
    task: str = "easy"

# --- In-Memory State ---
session_state = {}

def get_initial_state(task: str) -> Dict:
    if task == "easy":
        # Task 1: Turn off lights during the day
        return {
            "task": "easy",
            "step": 0,
            "time_of_day": "14:00 (Daytime)",
            "devices": [
                {"id": "light_living_room", "type": "light", "status": "on", "power": 60},
                {"id": "light_kitchen", "type": "light", "status": "on", "power": 60},
            ]
        }
    elif task == "medium":
        # Task 2: Set HVAC to Eco Mode (78F)
        return {
            "task": "medium",
            "step": 0,
            "time_of_day": "15:00",
            "devices": [
                {"id": "hvac_main", "type": "hvac", "status": "on", "temperature": 70.0, "power": 3500},
            ]
        }
    else:
        # Task 3 (Hard): Peak shaving. Turn off pool pump, set HVAC to 78.
        return {
            "task": "hard",
            "step": 0,
            "time_of_day": "18:00 (Peak Hours)",
            "devices": [
                {"id": "pool_pump", "type": "appliance", "status": "on", "power": 1500},
                {"id": "hvac_main", "type": "hvac", "status": "on", "temperature": 72.0, "power": 3000},
            ]
        }

def build_observation(state: Dict, feedback: str = "System ready.") -> Observation:
    devs = [DeviceState(id=d["id"], type=d["type"], status=d["status"], temperature=d.get("temperature")) for d in state["devices"]]
    total_power = sum(d["power"] for d in state["devices"] if d["status"] == "on")
    return Observation(time_of_day=state["time_of_day"], devices=devs, total_power_watts=total_power, feedback=feedback)

# --- Endpoints ---
@app.post("/reset", response_model=StepResult)
async def reset_env(req: ResetRequest = None):
    task_name = req.task if req else "easy"
    if task_name not in ["easy", "medium", "hard"]:
        task_name = "easy"
    
    global session_state
    session_state = get_initial_state(task_name)
    obs = build_observation(session_state)
    return StepResult(observation=obs, reward=0.0, done=False)

@app.get("/state", response_model=Observation)
async def get_state():
    if not session_state:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return build_observation(session_state)

@app.post("/step", response_model=StepResult)
async def step_env(action: Action):
    global session_state
    if not session_state:
        raise HTTPException(status_code=400, detail="Environment not reset.")
    
    session_state["step"] += 1
    feedback = f"Action '{action.command}' applied to '{action.device_id}'."
    reward = 0.0
    done = False
    
    # Apply action
    device = next((d for d in session_state["devices"] if d["id"] == action.device_id), None)
    if not device:
        return StepResult(observation=build_observation(session_state, "Error: Device not found"), reward=0.0, done=False, error="Device not found")

    if action.command in ["turn_on", "turn_off"]:
        device["status"] = action.command.split("_")[1]
    elif action.command == "set_temp" and action.value is not None and device["type"] == "hvac":
        device["temperature"] = action.value
    else:
        feedback = "Invalid command for device type."

    # Graders / Partial Reward Logic
    task = session_state["task"]
    if task == "easy":
        # Reward for turning off lights
        lights_on = sum(1 for d in session_state["devices"] if d["type"] == "light" and d["status"] == "on")
        reward = 1.0 - (lights_on / 2.0)
        if lights_on == 0:
            done = True
            
    elif task == "medium":
        # Reward based on proximity to 78F target
        hvac = session_state["devices"][0]
        dist = abs(78.0 - hvac["temperature"])
        reward = max(0.0, 1.0 - (dist / 8.0)) # 0.0 at 70F, 1.0 at 78F
        if hvac["temperature"] == 78.0:
            done = True
            
    elif task == "hard":
        # Reward = 0.5 for pump off, 0.5 for HVAC at 78F
        pump = next(d for d in session_state["devices"] if d["id"] == "pool_pump")
        hvac = next(d for d in session_state["devices"] if d["id"] == "hvac_main")
        
        r_pump = 0.5 if pump["status"] == "off" else 0.0
        dist = abs(78.0 - hvac["temperature"])
        r_hvac = max(0.0, 0.5 - (dist / 12.0))
        
        reward = r_pump + r_hvac
        if pump["status"] == "off" and hvac["temperature"] == 78.0:
            done = True

    # Timeout
    if session_state["step"] >= 8:
        done = True

    obs = build_observation(session_state, feedback)
    return StepResult(observation=obs, reward=reward, done=done)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
