"""Persistent, explicitly commanded robot session with camera and decision logs."""
import datetime
import json
import os
from pathlib import Path
import sys
import threading
import time

import cv2

ROOT = Path('outputs') / ('peg_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
ROOT.mkdir(parents=True)
lock = threading.Lock()
frames = {}
stop = threading.Event()
log = (ROOT / 'decisions.jsonl').open('a', buffering=1)


def record(value):
    log.write(json.dumps({'time': datetime.datetime.now().isoformat(), **value}, ensure_ascii=False) + '\n')
    log.flush()
    os.fsync(log.fileno())


def capture(index):
    cap = cv2.VideoCapture(index)
    writer = None
    try:
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        count = 0
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            if count < 10:
                continue
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(ROOT / f'camera{index}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 15, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Video writer failed')
            writer.write(frame)
            with lock:
                frames[index] = (time.time(), frame.copy())
            time.sleep(1 / 30)
    finally:
        cap.release()
        if writer:
            writer.release()


record({'command': 'camera_probe', 'summary': 'Identify views and record before any robot connection or movement.'})
threads = [threading.Thread(target=capture, args=(i,), daemon=True) for i in range(3)]
for thread in threads:
    thread.start()
arm = None
print('SESSION ' + str(ROOT.resolve()), flush=True)
try:
    for line in sys.stdin:
        try:
            cmd = json.loads(line)
            assert cmd.get('summary'), 'Decision summary required'
            record({'phase': 'before', **cmd})
            op = cmd['op']
            result = None
            if op == 'snapshot':
                with lock:
                    result = {}
                    for i, (stamp, frame) in frames.items():
                        path = ROOT / f'{cmd["label"]}_cam{i}.jpg'
                        cv2.imwrite(str(path), frame)
                        result[i] = {'path': str(path.resolve()), 'age': time.time() - stamp}
            elif op == 'connect':
                from lerobot.scripts.lerobot_so101_gui import ArmController
                arm = ArmController()
                result = arm.connect(cmd['port'])
            elif op == 'state':
                result = {'state': arm.get_state(), 'temperatures': arm.temperatures()}
            elif op in ('nudge', 'gripper'):
                with lock:
                    assert sum(time.time() - stamp < 2 for stamp, _ in frames.values()) >= 2, 'Need two live cameras'
                arm.check_temperatures()
                if op == 'nudge':
                    assert abs(cmd['amount']) <= (0.01 if cmd['axis'] in ('x', 'y', 'z') else 5)
                    result = arm.nudge(cmd['axis'], cmd['amount'])
                else:
                    result = arm.set_gripper(cmd['value'])
            elif op == 'stop':
                break
            else:
                raise ValueError(op)
            record({'phase': 'after', 'op': op, 'result': result})
            print(json.dumps(result), flush=True)
        except Exception as exc:
            record({'error': repr(exc)})
            print('ERROR ' + repr(exc), flush=True)
finally:
    if arm is not None:
        arm.disconnect()
    stop.set()
    for thread in threads:
        thread.join(timeout=5)
    record({'event': 'session_closed'})
    log.close()
