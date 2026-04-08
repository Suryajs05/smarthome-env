# Smart Home Energy Optimizer (OpenEnv Hackathon)

## Project Overview
This environment simulates a Smart Home Energy Management System. The goal is to provide a sandbox where AI agents can learn to balance inhabitant comfort with electrical efficiency. In this specific "Easy" task, the agent must identify when lights are left on during daylight hours and turn them off to conserve power.

## The Challenge
AI agents must process multi-modal state data, including:
- Temporal Data: The current time of day (Day vs. Night).
- Device States: Real-time status of multiple IoT devices (Living Room Light, Kitchen Light).
- Logic-Based Rewards: The agent only receives a maximum reward (+1.0) if it achieves the goal of turning off all unnecessary loads without interfering with
  required night-time lighting.

## Environment Description
A real-world simulation of an automated home energy management system. The agent must read the state of various appliances, understand peak hours, and issue commands to optimize energy usage while respecting comfort boundaries.

## Action Space
The agent outputs JSON matching the following schema:
- `device_id` (str): The target device.
- `command` (str): "turn_on", "turn_off", or "set_temp".
- `value` (float, optional): Target value for temperature controls.

## Observation Space
JSON containing:
- `time_of_day`: Current simulated time.
- `devices`: Array of objects showing ID, type, status, and temperature.
- `total_power_watts`: Active power draw.
- `feedback`: Execution result of the last command.

## Setup Instructions
1. Install requirements: `pip install -r requirements.txt`
2. Run the environment locally: `uvicorn server.main:app --port 7860`
3. In a separate terminal, export required variables:
   ```bash
   export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
   export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
   export HF_TOKEN="your_hf_token"
   export ENV_URL="http://localhost:7860"
