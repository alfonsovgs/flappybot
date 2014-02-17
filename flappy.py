from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager, BaseProxy
import Quartz
import Quartz.CoreGraphics as CG
import cv2
import numpy
import pickle
import time


kPaddingTop = 50
kFieldHeight = 370
kBirdX = 105
kBirdWidth = 20
kPipePixelThreshold = 100
kPipeMaxWidth = 70
kPipeMinWidth = 3
kPipeMinOpening = 60
kBeakMinColor = numpy.array([1, 100, 100], numpy.uint8)
kBeakMaxColor = numpy.array([20, 255, 255], numpy.uint8)
kBeakMinSize = 3
kBeakMaxLength = 10
kClimbAltGain = 40.0
kClimbDuration = 0.265

def tap_main(tap_queue):
  while True:
    tap_queue.get() # blocking
    fd = open("/dev/tty.usbserial-A6008aIZ", "w")
    fd.write("1")
    fd.close()
    #print "Tap!", time.time()


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
    self.last_tap_ts = None
    cv2.namedWindow('monitor')
    cv2.namedWindow('field')
    self.tap_queue = tap_queue

  @classmethod
  def get_window_id(cls):
    windows = Quartz.CGWindowListCopyWindowInfo(
        CG.kCGWindowListOptionAll,
        CG.kCGNullWindowID)
    for window in windows:
      if ('wickedman' in window.get(CG.kCGWindowName, '')
          and 'X-Mirage' in window.get(CG.kCGWindowOwnerName, '')) \
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
    self.last_tap_ts = time.time()

  def capture_images(self):
    self.capture_window(self.capture_img, self.window_id)
    self.display_image = cv2.cvtColor(self.capture_img, cv2.COLOR_RGB2BGR)
    self.bird_image = self.display_image[kPaddingTop:kPaddingTop + kFieldHeight,
        kBirdX - kBirdWidth / 2 : kBirdX + kBirdWidth / 2]
    blue, _, _ = cv2.split(self.display_image)
    _, bw_image = cv2.threshold(blue, 140, 255, cv2.THRESH_BINARY)
    self.field_image = bw_image[kPaddingTop:kPaddingTop + kFieldHeight, kBirdX:]

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
    cv2.imshow('bird filtered', threshed)
    return None if best is None else best[0] + 8  # Aim for the center

  def get_state(self):
    bird_y = self._find_bird()
    pipe_x, pipe_width, pipe_top_h, pipe_bottom_h = self._find_pipe()
    return State(bird_y, pipe_x, pipe_width, pipe_top_h, pipe_bottom_h)

  def refresh_monitors(self):
    cv2.imshow('monitor', self.display_image)
    cv2.imshow('field', self.field_image)

  def highlight_state(self, state):
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


class FlightMode:
  FLOAT = 1
  CLIMB = 2
  DIVE = 3


def autopilot_main(published_state, tap_queue):
  last_tap_ts = [time.time()]
  last_mode = [None]
  def autopilot_logic(last_state):
    do_tap = False
    since_tap = time.time() - last_tap_ts[0]
    if last_state:
      target = last_state.pipe_top_h + last_state.opening_h() / 2
      dive_x = since_tap - kClimbDuration
      state_x = last_state.timestamp - last_tap_ts[0] - kClimbDuration
      if dive_x >= 0 and state_x >= 0:
        est_y = last_state.bird_y + 600.0 * (dive_x**2 - state_x**2)
      else:
        est_y = last_state.bird_y
    else:
      target = 200.0
      est_y = last_state.bird_y
      dive_x, state_x = 0, 0

    mode = FlightMode.FLOAT
    if last_state.bird_y:
      delta = last_state.bird_y - target
      if delta >= 60:
        mode = FlightMode.CLIMB
      elif delta <= -60 and since_tap >= kClimbDuration:
        mode = FlightMode.DIVE
      else:
        mode = FlightMode.FLOAT
    mode = FlightMode.FLOAT

    if mode == FlightMode.FLOAT:
      multiplier = 1.0
      if est_y:
        if est_y > target:
          multiplier = 0.8
        elif est_y < target:
          multiplier = 1.2
      if since_tap > 0.578 * multiplier:
        do_tap = True
    elif mode == FlightMode.CLIMB:
      if since_tap > 0.2:
        do_tap = True
    elif mode == FlightMode.DIVE:
      if dive_x >= 0 and state_x >= 0:
        #if not throttle('blabla', 10):
        #  print "Dive est y", est_y, dive_x, state_x, time.time()
        if est_y >= target:
          do_tap = True

    if mode != last_mode[0]:
      print mode, target, last_state.bird_y, do_tap
      last_mode[0] = mode

    if do_tap:
      tap_queue.put_nowait(True)
      last_tap_ts[0] = time.time()

  while True:
    last_state = published_state._getvalue()
    #if not throttle('autopilot monitor', 1):
    #  print '>', last_state
    autopilot_logic(last_state)
    time.sleep(throttle('autopilot', 30))


def game_main(tap_queue, published_state):
  game = GameUI(tap_queue)
  state_log = [(None, [])]
  last_state = None
  while True:
    if not throttle('capture', 20):
      game.capture_images()
      last_state = game.get_state()
      state_log[-1][1].append(last_state)
    if not throttle('monitor', 10):
      game.highlight_state(last_state)
      game.refresh_monitors()
    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # Esc
      break
    elif key == 32: # Space
      if not throttle('tap', 100):
        game.tap()
        state_log.append((game.last_tap_ts, []))
      else:
        print "Declined keyboard tap"
    published_state.update(last_state)
    time.sleep(throttle('loop', 30))
  open("/tmp/flappybot.log", "w").write(pickle.dumps(state_log))


class StateProxy(BaseProxy):
  _exposed_ = ('update',)
  def update(self, new_state):
    self._callmethod('update', (new_state,))


class FlappyManager(BaseManager):
  pass

FlappyManager.register('State', State, proxytype=StateProxy)
FlappyManager.register('Queue', Queue)



def main():
  process_manager = FlappyManager()
  process_manager.start()
  tap_queue = process_manager.Queue()
  published_state = process_manager.State()
  tap_proc = Process(target=tap_main, args=(tap_queue,))
  tap_proc.start()
  autopilot_proc = Process(target=autopilot_main,
      args=(published_state, tap_queue,))
  autopilot_proc.start()
  game_main(tap_queue, published_state)
  autopilot_proc.terminate()
  tap_proc.terminate()

if __name__ == '__main__':
  main()
