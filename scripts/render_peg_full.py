"""Render complete paired camera recordings with bilingual decision summaries."""
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1] / 'outputs/peg_20260917_023406'
ASSETS = ROOT / 'full_video_assets'
ASSETS.mkdir(exist_ok=True)
records = [json.loads(line) for line in (ROOT / 'decisions.jsonl').read_text().splitlines()]
start = datetime.fromisoformat(records[0]['time'])
end = datetime.fromisoformat(records[-1]['time'])
wall = (end - start).total_seconds()

def duration(path):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(path)]))

d0, d1 = [duration(ROOT / f'camera{i}.avi') for i in range(2)]
total = max(d0, d1)
frames = math.ceil(total * 15)
total = frames / 15

# Summaries of the saved observations/actions, not private internal reasoning.
phases = [
('02:34:06', '检查场景与连接', '确认两个视角中的圆柱、孔座和初始间隙；开始连续记录，再连接机械臂。', 'Inspect scene and connect', 'Identify the peg, socket and initial clearance in both views. Start continuous recording before connecting the arm.'),
('02:34:30', '抬升并调整姿态', '空夹爪分步抬升，检查周围间隙；逐步转为向下抓取姿态。', 'Raise and orient the gripper', 'Raise the empty gripper in small steps, checking clearance, then rotate toward a downward grasp.'),
('02:37:00', '确认运动方向', '用小幅 X / Y 移动观察图像变化，张开夹爪，建立接近方向。', 'Check motion directions', 'Use small X / Y moves to observe image displacement. Open the jaws and establish the approach direction.'),
('02:37:50', '从两个视角对准圆柱', '横向接近并修正偏差；先保持高度，确认夹爪中心与圆柱的位置关系。', 'Align over the peg', 'Approach laterally and correct the offset using both views. Maintain height while checking jaw-to-peg alignment.'),
('02:40:00', '小步下降接近', '确认固定侧夹爪避开圆柱顶部；分步下降，观察圆柱是否倾斜或移动。', 'Descend in small steps', 'Check that the fixed jaw clears the peg top. Descend gradually while watching for peg tilt or displacement.'),
('02:41:50', '第一次抓取', '逐步闭合夹爪，再小幅试提；以圆柱底部是否离开桌面判断抓取效果。', 'First grasp attempt', 'Close the jaws incrementally and make a small test lift. Check whether the peg base leaves the table.'),
('02:43:00', '试提后发现抓取不稳', '圆柱未稳定随夹爪上升；收紧后仍有偏夹和倾斜，需要放下重新对中。', 'Unstable grasp detected', 'The peg does not rise reliably with the jaws. A tighter grip still shows edge contact and tilt; lower it and re-center.'),
('02:43:45', '放下并重新对中', '降低并松开圆柱，确认恢复直立；抬高夹爪后修正横向位置。', 'Release and re-center', 'Lower and release the peg, checking that it stands upright. Raise the jaws before correcting lateral alignment.'),
('02:44:45', '复核温度读数', '出现间歇性偏高的肘关节温度读数，暂停并重复检查；后续读数恢复后继续。', 'Recheck temperature readings', 'Intermittent high elbow-temperature samples trigger pauses and repeated checks. Continue after subsequent readings return lower.'),
('02:45:25', '重新下降与抓取', '按修正后的中心下降，再逐步闭合夹爪；试提验证圆柱是否稳定离开桌面。', 'Retry the grasp', 'Descend at the revised center and close the jaws gradually. Test-lift to verify that the peg clears the table securely.'),
('02:46:49', '确认抓取并抬至搬运高度', '第二次抓取使圆柱离开桌面；进一步抬高，检查底部间隙和抓取稳定性。', 'Confirm grasp and raise for transfer', 'The revised grasp lifts the peg off the table. Raise further and check bottom clearance and grasp stability.'),
('02:47:20', '向孔座搬运', '保持圆柱悬空，分步横移；每段移动后检查抓取状态与孔口偏差。', 'Transfer toward the socket', 'Keep the peg suspended and move laterally in steps, checking grip stability and the remaining socket offset.'),
('02:48:35', '关节限位保护与路径调整', '一次向前移动因腕关节限位被拒绝；随后调整横向路径，避免继续过度伸展。', 'Joint-limit protection and route adjustment', 'A forward move is rejected by the wrist joint limit. Adjust the lateral route to avoid further excessive extension.'),
('02:50:20', '孔口上方精调', '通过两个视角修正深度和横向误差；接近孔口时缩小移动步长。', 'Fine alignment above the opening', 'Correct depth and lateral offsets from both views. Reduce step size as the peg approaches the socket rim.'),
('02:51:45', '开始毫米级插入', '以 1–2 毫米的小步下降；持续观察孔座位移、圆柱倾斜和边缘接触。', 'Begin millimeter-scale insertion', 'Lower in 1–2 mm steps, watching for socket movement, peg tilt and contact with the rim.'),
('02:53:35', '继续进入孔内', '图像中圆柱逐步进入孔口；继续小步下降，检查是否卡住或推动孔座。', 'Continue entry into the socket', 'The peg appears to enter the opening. Continue with small downward steps while checking for wedging or socket displacement.'),
('02:54:35', '放松夹持，让圆柱落座', '逐步张开夹爪，让圆柱在重力作用下落座；检查孔座位置和圆柱稳定性。', 'Relax the grip and let the peg seat', 'Open the jaws incrementally so the peg can settle under gravity. Check socket position and peg stability.'),
('02:55:16', '松爪后确认插入完成', '夹爪与圆柱分离后，圆柱仍稳定留在孔座内；撤离夹爪，再次记录完成状态。', 'Verify insertion after release', 'With the jaws separated, the peg remains seated in the socket. Withdraw the gripper and document the completed placement.'),
('02:55:56', '撤回并停放机械臂', '空夹爪远离已插入的圆柱，分步降低至停放位置；保留全部收尾录像。', 'Retract and park the arm', 'Move the empty jaws away from the seated peg and lower gradually to a parked pose. Keep the entire shutdown sequence in the recording.'),
('02:57:36', '完成状态与断开连接', '插入结果保持稳定；机械臂已停放，随后断开连接并结束录像。', 'Final state and disconnection', 'The inserted peg remains in place. The arm is parked, then disconnected as the recording ends.'),
]
fontpath = '/System/Library/Fonts/Hiragino Sans GB.ttc'
font = ImageFont.truetype(fontpath, 31)
titlefont = ImageFont.truetype(fontpath, 36)
small = ImageFont.truetype(fontpath, 23)

def wrap(draw, value, maxwidth):
    units = value.split(' ') if ' ' in value and value.isascii() else list(value)
    joiner = ' ' if value.isascii() else ''
    lines, current = [], ''
    for unit in units:
        candidate = current + (joiner if current else '') + unit
        if draw.textlength(candidate, font=font) > maxwidth and current:
            lines.append(current)
            current = unit
        else:
            current = candidate
    return lines + [current]

times = [max(0, (datetime.fromisoformat('2026-09-17T' + p[0]) - start).total_seconds()) / wall * total for p in phases]
times[0] = 0
for lang in ('zh', 'en'):
    entries = []
    for i, p in enumerate(phases):
        title, summary = p[1:3] if lang == 'zh' else p[3:5]
        img = Image.new('RGB', (1920, 220), '#101925')
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 1920, 3), fill='#55c9be')
        header = 'CAMERA 0                                          CAMERA 1'
        note = '完整录像 · 双视角近似同步 · 基于操作日志的决策摘要' if lang == 'zh' else 'FULL RECORDING | Approximate dual-camera sync | Summaries from the action log'
        draw.text((28, 12), 'CAMERA 0  /  CAMERA 1', font=small, fill='#9aabba')
        draw.text((630, 12), note, font=small, fill='#9aabba')
        draw.text((28, 48), f'{i+1:02d} / {len(phases):02d}   {title}', font=titlefont, fill='#55c9be')
        for j, line in enumerate(wrap(draw, summary, 1855)):
            draw.text((28, 101 + j * 43), line, font=font, fill='#ffffff')
        path = ASSETS / f'{lang}_{i:02d}.png'
        img.save(path)
        nexttime = times[i+1] if i+1 < len(times) else total
        entries.extend([f"file '{path}'", f'duration {nexttime-times[i]:.6f}'])
    entries.append(f"file '{path}'")
    (ASSETS / f'{lang}.ffconcat').write_text('ffconcat version 1.0\n' + '\n'.join(entries) + '\n')

graph = (f'[0:v]setpts=(PTS-STARTPTS)*{total/d0:.12f},scale=960:540,fps=15,setsar=1[a];'
         f'[1:v]setpts=(PTS-STARTPTS)*{total/d1:.12f},scale=960:540,fps=15,setsar=1[b];'
         '[a][b]hstack=inputs=2,pad=1920:760:0:0:color=0x101925,split=2[c][d];'
         '[c][2:v]overlay=0:540:repeatlast=1,format=yuv420p[zh];'
         '[d][3:v]overlay=0:540:repeatlast=1,format=yuv420p[en]')
cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-i', str(ROOT/'camera0.avi'), '-i', str(ROOT/'camera1.avi')]
for lang in ('zh', 'en'):
    cmd += ['-f', 'concat', '-safe', '0', '-i', str(ASSETS/f'{lang}.ffconcat')]
cmd += ['-filter_complex_threads', '4', '-filter_complex', graph]
for lang in ('zh', 'en'):
    cmd += ['-map', f'[{lang}]', '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p', '-r', '15', '-frames:v', str(frames), '-movflags', '+faststart', str(ROOT/f'peg_full_camera01_{lang}.mp4')]
(ASSETS/'render_manifest.json').write_text(json.dumps({'input_durations': [d0,d1], 'output_duration':total, 'frames':frames,'sync':'Linear duration normalization; no per-frame capture timestamps available.', 'phases':phases, 'command':cmd}, ensure_ascii=False, indent=2))
subprocess.run(cmd, check=True)
