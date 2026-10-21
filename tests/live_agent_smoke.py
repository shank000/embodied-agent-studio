"""Explicit opt-in: one real Codex vision call and actual MuJoCo TCP motion."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.app import Worker
from server.agent import AgentController, AgentStart


async def main():
    worker = Worker(); worker.thread.start()
    try:
        if not await asyncio.to_thread(worker.ready.wait, 30) or worker.failure:
            raise RuntimeError(worker.failure or 'engine timeout')
        agent = AgentController(worker)
        await agent.start(AgentStart(task='轻量测试：看三路图像，末端沿世界坐标 Z 轴抬高 0.02 米，保持当前 X、Y、四元数姿态和夹爪开度。本轮请给出这个小动作。', one_round=True))
        await agent.task
        result = agent.snapshot()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result['status'] != 'round_complete' or not result['history'] or result['history'][0].get('result', {}).get('status') != 'complete':
            raise RuntimeError('Real model / actual motion acceptance failed; inspect result above.')
    finally:
        worker.shutdown.set(); await asyncio.to_thread(worker.thread.join, 5)


if __name__ == '__main__': asyncio.run(main())
