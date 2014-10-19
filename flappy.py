from multiprocessing import Process, Queue, Value
from multiprocessing.managers import BaseManager, BaseProxy
import Quartz
import Quartz.CoreGraphics as CG
import cv2
import math
import numpy
import pickle
import serial
import time


kPaddingTop = 40
kFieldHeight = 370
kBirdX = 105
kBirdWidth = 20
kPipePixelThreshold = 100
kPipeMaxWidth = 70
kPipeMinWidth = 3
kPipeMinOpening = 90
kBeakMinColor = numpy.array([1, 100, 100], numpy.uint8)
kBeakMaxColor = numpy.array([20, 255, 255], numpy.uint8)
kBeakMinSize = 3
kBeakMaxLength = 10
kClimbAltGain = 40
kClimbDuration = 0.18
kSpeed = 131.31
kLatencyAdjust = 0.3
kTapCycle = 0.595


class State(object):
  def __init__(self, bird_y=None, pipe_x=None, pipe_width=None,
      pipe_top_h=None, pipe_bottom_h=None):
    self.bird_y = bird_y
    self.pipe_x = pipe_x
    self.pipe_width = pipe_width
    self.pipe_top_h = pipe_top_h
    self.pipe_bottom_h = pipe_bottom_h
    self.timestamp = time.time()

  def __nonzero__(self):
    return self.bird_y is not None and self.pipe_x is not None

  def __repr__(self):
    d = dict(bird_y=self.bird_y, pipe_x=self.pipe_x,
        pipe_width=self.pipe_width, pipe_top_h=self.pipe_top_h,
        pipe_bottom_h=self.pipe_bottom_h, timestamp=self.timestamp)
    return ' '.join('%s=%s' % (k, v) for (k, v) in d.items())

  def opening_h(self):
    if not self:
      raise ValueError()
    return kFieldHeight - self.pipe_top_h - self.pipe_bottom_h

  def update(self, other_state):
    self.bird_y = other_state.bird_y
    self.pipe_x = other_state.pipe_x
    self.pipe_width = other_state.pipe_width
    self.pipe_top_h = other_state.pipe_top_h
    self.pipe_bottom_h = other_state.pipe_bottom_h
    self.timestamp = other_state.timestamp


class GameUI(object):
  def __init__(self, tap_queue):
    self.window_id = self.get_window_id()
    self.capture_img = numpy.zeros((568, 320, 4), numpy.uint8)
    cv2.namedWindow('monitor')
    cv2.namedWindow('field')
    self.tap_queue = tap_queue

  @classmethod
  def get_window_id(cls):
    windows = Quartz.CGWindowListCopyWindowInfo(
        CG.kCGWindowListOptionAll,
        CG.kCGNullWindowID)
    for window in windows:
      if ('wickedman' in window.get(CG.kCGWindowName, '')) \
          or 'test_screenshot' in window.get(CG.kCGWindowName, ''):
        return window[CG.kCGWindowNumber]
    raise Exception("Couldn't find capture window")

  @classmethod
  def capture_window(cls, cv_image, window_id):
    height, width, num_components = cv_image.shape
    image = CG.CGWindowListCreateImageFromArray(
        CG.CGRectNull,
        [window_id],
        CG.kCGWindowImageBoundsIgnoreFraming)
    color_space = CG.CGColorSpaceCreateDeviceRGB()
    context_ref = CG.CGBitmapContextCreate(
        cv_image.data,
        width,
        height,
        8,
        width * num_components,
        color_space,
        CG.kCGImageAlphaNoneSkipLast | CG.kCGBitmapByteOrderDefault)
    CG.CGContextDrawImage(
        context_ref,
        CG.CGRectMake(0, 14, width, height),
        image)

  def tap(self):
    self.tap_queue.put_nowait(True)

  def capture_images(self):
    self.capture_window(self.capture_img, self.window_id)
    self.display_image = cv2.cvtColor(self.capture_img, cv2.COLOR_RGB2BGR)
    self.bird_image = self.display_image[kPaddingTop:kPaddingTop + kFieldHeight,
        kBirdX - kBirdWidth / 2 : kBirdX + kBirdWidth / 2]
    blue, _, _ = cv2.split(self.display_image)
    _, bw_image = cv2.threshold(blue, 140, 255, cv2.THRESH_BINARY)
    self.field_image = bw_image[kPaddingTop:kPaddingTop + kFieldHeight,
        kBirdX - kBirdWidth:]

  def _find_pipe(self):
    havg = cv2.reduce(self.field_image, 0, cv2.cv.CV_REDUCE_AVG)[0]
    for x in xrange(len(havg)):
      if havg[x] > kPipePixelThreshold:
        continue
      # Find width
      width = 0
      while (x + width < len(havg)) and (havg[x + width] <= kPipePixelThreshold):
        width += 1
      if not (kPipeMinWidth <= width <= kPipeMaxWidth):
        x += width
        continue
      # Find opening
      pipe_slice = self.field_image[:, x:x + width]
      vavg = cv2.reduce(pipe_slice, 1, cv2.cv.CV_REDUCE_AVG)
      def find_first_gap_row(vavg):
        for y in xrange(len(vavg)):
          if vavg[y][0] > kPipePixelThreshold:
            return y
        return None
      top_h = find_first_gap_row(vavg)
      bottom_h = find_first_gap_row(vavg[::-1])
      if top_h is None or bottom_h is None:
        continue
      if len(vavg) - top_h - bottom_h < kPipeMinOpening:
        continue
      return x - 2, width + 4, top_h, bottom_h
    return None, None, None, None

  def _find_bird(self):
    hsv_image = cv2.cvtColor(self.bird_image, cv2.COLOR_BGR2HSV)
    threshed = cv2.inRange(hsv_image, kBeakMinColor, kBeakMaxColor)
    v = [p[0] >= 10 for p in cv2.reduce(threshed, 1, cv2.cv.CV_REDUCE_AVG)]
    best = None
    for y in xrange(len(v)):
      for height in xrange(min(kBeakMaxLength, len(v) - y)):
        if not v[y + height]:
          break
        if height >= kBeakMinSize:
          if best is None or best[1] < height:
            best = (y, height)
    # cv2.imshow('bird filtered', threshed)
    return None if best is None else best[0] + 8  # Aim for the center

  def get_state(self):
    bird_y = self._find_bird()
    pipe_x, pipe_width, pipe_top_h, pipe_bottom_h = self._find_pipe()
    return State(bird_y, pipe_x, pipe_width, pipe_top_h, pipe_bottom_h)

  def refresh_monitors(self):
    cv2.imshow('monitor', self.display_image)
    cv2.imshow('field', self.field_image)

  def highlight_state(self, state):
    self._highlight_grid()
    if state:
      self._highlight_pipe(state)
    if state.bird_y:
      cv2.circle(self.display_image, (kBirdX, kPaddingTop + state.bird_y),
          5, (255, 0, 0), 3)

  def _highlight_pipe(self, state):
    cv2.rectangle(self.display_image,
        (state.pipe_x + kBirdX, 0),
        (state.pipe_x + state.pipe_width + kBirdX, kPaddingTop + state.pipe_top_h),
        (0, 0, 255), 2, 8)
    cv2.rectangle(self.display_image,
        (state.pipe_x + kBirdX, kPaddingTop + kFieldHeight - state.pipe_bottom_h),
        (state.pipe_x + state.pipe_width + kBirdX, kPaddingTop + kFieldHeight),
        (0, 0, 255), 2, 8)
    opening_h = kFieldHeight - state.pipe_top_h - state.pipe_bottom_h
    cv2.line(self.display_image,
        (state.pipe_x + kBirdX - 5, kPaddingTop + state.pipe_top_h + opening_h / 2),
        (state.pipe_x + state.pipe_width + kBirdX + 5, kPaddingTop + state.pipe_top_h + opening_h / 2),
        (0, 255, 0), 3, 8)

  def _highlight_grid(self):
    for y in range(50, kFieldHeight, 50):
      cv2.line(self.display_image,
          (0, kPaddingTop + y), (320, kPaddingTop + y),
          (200, 200, 200), 1 + int(y % 100 == 0), 8)
      cv2.line(self.display_image,
          (y, kPaddingTop), (y, kPaddingTop + kFieldHeight),
          (200, 200, 200), 1 + int(y % 100 == 0), 8)

  def highlight_est_trajectory(self, last_state, last_tap_ts, tap_y):
    points = []
    t1 = last_state.timestamp - last_tap_ts
    for ti in range(0, 40):
      t = ti * 0.05
      y = self.est_bird_y(last_tap_ts, tap_y, last_tap_ts + t)
      points.append((int((t - t1) * kSpeed + kBirdX), int(y + kPaddingTop)))
    def plot(p):
      for i in range(1, len(p)):
        cv2.line(self.display_image, p[i-1], p[i],
            (0, 250, 0), 4, 8)
    plot(points)

  def __del__(self):
    cv2.destroyAllWindows()


def throttle(key, hz, ts_dict=dict()):
  now = time.time()
  if key not in ts_dict:
    ts_dict[key] = now
    return 0.0
  left = ts_dict[key] + 1.0/hz - now
  if left < 0:
    ts_dict[key] = now
  return max(0.0, left)


def autopilot_main(published_state, tap_queue, last_tap_ts):
  tap_y = [None]
  addtl_float_multiplier = [1.0]

  def autopilot(state_buffer, last_state):
    if not state_buffer:
      print "WARNING: State buffer empty!"
      state_buffer = [last_state]
    if last_state.bird_y is None:
      return
    ys = sorted([s.bird_y for s in state_buffer if s.bird_y is not None])
    med_y = ys[len(ys) / 2]
    tap_ts = last_tap_ts.value
    since_tap = time.time() - tap_ts
    if last_state:
      target = last_state.pipe_top_h + last_state.opening_h() * 0.5
    else:
      target = 200.0

    def _float_adjust(delta):
      if delta < 0:
        multiplier = 0.85 * addtl_float_multiplier[0]
      elif delta > 10:
        multiplier = 1.15
      else:
        multiplier = 1
      if since_tap > kTapCycle * multiplier:
        tap_queue.put_nowait(True)
        addtl_float_multiplier[0] = 1.0

    def _fall_position_est(y0, t):
      kA = 458.314
      kB = 85.0
      kC = -1.43
      y = kA * t**2 + kB * t + kC
      return y0 + int(y)

    def _dive_to(target):
      points = [(s.timestamp, s.bird_y) for s in state_buffer \
          if s.bird_y is not None]
      if len(points) < 2:
        return
      # find apex
      i = len(points) - 1
      while i > 0 and points[i - 1][1] < points[i][1]:
        i -= 1
      est_y = _fall_position_est(points[i][1],
          time.time() - points[i][0] + kLatencyAdjust)
      #print "DIVE est y", est_y, " instead of ", last_state.bird_y, "target", \
      #    target
      if est_y > target - 15 and since_tap >= kTapCycle:
        tap_queue.put_nowait(True)

    def _climb(height):
      for i in range(int(math.ceil((height - 30) / kClimbAltGain))):
        tap_queue.put_nowait(True)
        time.sleep(kClimbDuration)
      time.sleep(kLatencyAdjust)

    if med_y < target - 70:
       _dive_to(target)
    elif med_y > target + 50:
      _climb(med_y - target)
      # addtl_float_multiplier[0] = 0.9
    else:
      _float_adjust(target - med_y)

  state_buffer = []
  while True:
    last_state = published_state._getvalue()
    now = time.time()
    state_buffer.append(last_state)
    while state_buffer and state_buffer[0].timestamp < now - kTapCycle:
      state_buffer = state_buffer[1:]
    autopilot(state_buffer, last_state)
    if not throttle('autopilot monitor', 10):
      print '>', last_state
    time.sleep(throttle('autopilot', 30))


def game_main(tap_queue, published_state, last_tap_ts):
  game = GameUI(tap_queue)
  state_log = [(None, [])]
  last_state = None
  while True:
    if not throttle('capture', 60):
      game.capture_images()
      last_state = game.get_state()
      state_log[-1][1].append(last_state)
    if not throttle('monitor', 0.1):
      game.highlight_state(last_state)
      game.refresh_monitors()
    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # Esc
      break
    elif key == 32: # Space
      if not throttle('tap', 120):
        game.tap()
        state_log.append((time.time(), []))
      else:
        print "Declined keyboard tap"
    published_state.update(last_state)
    time.sleep(throttle('loop', 60))
  open("/tmp/flappybot.log", "w").write(pickle.dumps(state_log))


def tap_main(tap_queue, last_tap_ts):
  ser = serial.Serial('/dev/tty.usbmodemfd121', 115200)
  while True:
    tap_queue.get()
    ser.write("1")
    last_tap_ts.value = time.time()
    print "Tap!", last_tap_ts.value
  ser.close()


class StateProxy(BaseProxy):
  _exposed_ = ('update',)
  def update(self, new_state):
    self._callmethod('update', (new_state,))


class FlappyManager(BaseManager):
  pass

FlappyManager.register('State', State, proxytype=StateProxy)


def main():
  process_manager = FlappyManager()
  process_manager.start()
  tap_queue = Queue()
  published_state = process_manager.State()
  last_tap_ts = Value('d', time.time())
  tap_proc = Process(target=tap_main, args=(tap_queue, last_tap_ts))
  tap_proc.start()
  autopilot_proc = Process(target=autopilot_main,
      args=(published_state, tap_queue, last_tap_ts))
  autopilot_proc.start()
  game_main(tap_queue, published_state, last_tap_ts)
  autopilot_proc.terminate()
  tap_proc.terminate()

if __name__ == '__main__':
  main()
