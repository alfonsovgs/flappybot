"""
Throw-away script to analyze flappy state logs.
"""

from itertools import ifilter
import pickle
from pylab import *
import sys

from flappy import State


def analyze_gravity(state_log):
  dec_sequences = [[]]
  inc_sequences = [[]]
  for i in xrange(1, len(state_log)):
    crt, last = state_log[i], state_log[i - 1]
    if not crt or not last:
      if inc_sequences[-1]:
        inc_sequences.append([])
      if dec_sequences[-1]:
        dec_sequences.append([])
      continue
    if crt.bird_y < last.bird_y:
      if not dec_sequences[-1]:
        dec_sequences[-1].append(last)
      dec_sequences[-1].append(crt)
      if inc_sequences[-1]:
        inc_sequences.append([])
    elif crt.bird_y > last.bird_y:
      if not inc_sequences[-1]:
        inc_sequences[-1].append(last)
      inc_sequences[-1].append(crt)
      if dec_sequences[-1]:
        dec_sequences.append([])
  def format_seq(seq):
    x, y = [], []
    for s in seq:
      x.append(s.timestamp - seq[0].timestamp)
      y.append(-s.bird_y + seq[0].bird_y)
    plot(x, y)
  def format_sseq(sseq):
    sseq = [s for s in sseq if len(s) > 3]
    for seq in sseq:
      format_seq(seq)
  def plot_gravity_est():
    xs = [x * 0.05 for x in range(20)]
    ys = []
    for x in xs:
      ys.append(-640.0 * (x**2))
    plot(xs, ys, linestyle='dashed')
  subplot(1, 3, 1)
  format_sseq(inc_sequences)
  xlabel('time')
  ylabel('bird y')
  title('descent')
  plot_gravity_est()
  subplot(1, 3, 2)
  title('climb')
  xlabel('time')
  ylabel('bird y')
  format_sseq(dec_sequences)


def analyze_pipes(state_log):
  sequences = [[]]
  for i in xrange(1, len(state_log)):
    crt, last = state_log[i], state_log[i - 1]
    if not crt or not last:
      if sequences[-1]:
        sequences.append([])
      continue
    step = crt.pipe_x + crt.pipe_width - last.pipe_x - last.pipe_width
    if step > 100 or step >= 0:
      if sequences[-1]:
        sequences.append([])
      continue
    sequences[-1].append(crt)
  def format_seq(seq):
    x, y = [], []
    for s in seq:
      x.append(s.timestamp - seq[0].timestamp)
      y.append(s.pipe_x + s.pipe_width - seq[0].pipe_x - seq[0].pipe_width)
    plot(x, y)
  def format_sseq(sseq):
    sseq = [s for s in sseq if len(s) > 3]
    for seq in sseq:
      format_seq(seq)
  subplot(1, 3, 3)
  format_sseq(sequences)
  xlabel('time')
  ylabel('pipe x')
  title('distance to pipe')


def main(argv):
  log = pickle.loads(open(argv[1], "r").read())
  concat_log = []
  for ts, s in log:
    concat_log += s
  analyze_gravity(concat_log)
  analyze_pipes(concat_log)
  show()

if __name__ == '__main__':
  main(sys.argv)
