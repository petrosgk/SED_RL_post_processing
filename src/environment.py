import numpy as np
import tensorforce
import sed_eval
import hparams
from scipy.signal import medfilt


class SEDEnvironment(tensorforce.environments.Environment):
  def __init__(self, data, reference_data, max_length, evaluate=False):
    self.data = data
    self.reference_data = reference_data
    self.max_length = max_length
    self.segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(hparams.classes)
    self.event_based_metrics = sed_eval.sound_event.EventBasedMetrics(hparams.classes)
    self.evaluate = evaluate
    self.file_idx = 0
    self.predictions = []
    self.best_event_based_f1_score = 0
    self.best_segment_based_f1_score = 0
    super(SEDEnvironment, self).__init__()

  def states(self):
    return {'type': 'float', 'shape': (self.max_length, len(hparams.classes)), 'min_value': 0.0, 'max_value': 1.0}

  def actions(self):
      if hparams.class_dependent:
        num_classes = len(hparams.classes)
      else:
        num_classes = 1
      return {'window_sizes': {'type': 'int', 'shape': num_classes, 'num_values': len(hparams.window_sizes)},
              'thresholds': {'type': 'float', 'shape': num_classes, 'min_value': hparams.threshold_range[0], 'max_value': hparams.threshold_range[1]}}

  def max_episode_timesteps(self):
    return len(self.data)

  def reset(self):
    self.predictions = []
    self.event_based_metrics.reset()  # Reset event-based metrics
    self.segment_based_metrics.reset()  # Reset segment-based metrics
    self.file_idx = 0  # Reset file index
    return dict(state=self.data[0])  # Return first file

  def get_event_list(self, window_sizes, thresholds):
    event_list = []
    file = self.data[self.file_idx]
    num_classes = len(hparams.classes)
    num_frames = len(file)
    for class_idx in range(num_classes):
      if hparams.class_dependent:
        file[:, class_idx] = medfilt(file[:, class_idx], kernel_size=window_sizes[class_idx])
        threshold = thresholds[class_idx]
      else:
        file[:, class_idx] = medfilt(file[:, class_idx], kernel_size=window_sizes[0])
        threshold = thresholds[0]
      frame_idx = 0
      while frame_idx < num_frames:
        if file[frame_idx][class_idx] >= threshold:
          onset_frame_idx = frame_idx
          while (frame_idx < num_frames) and (file[frame_idx][class_idx] >= threshold):
            frame_idx += 1
          offset_frame_idx = frame_idx - 1
          onset = (onset_frame_idx * hparams.hop_length) / hparams.sample_rate
          offset = (offset_frame_idx * hparams.hop_length) / hparams.sample_rate
          event_label = hparams.classes[class_idx]
          event_list.append(
            {'onset': float(onset), 'offset': float(offset), 'event_label': event_label}
          )
        frame_idx += 1
    event_list = sorted(event_list, key=lambda x: x['onset'])
    return event_list

  def compute_average_class_wise_f_measure(self):
    event_based_f1_score_per_class = []
    segment_based_f1_score_per_class = []
    for class_label in hparams.classes:
      event_based_class_f1_score = self.event_based_metrics.class_wise_f_measure(class_label)['f_measure']
      event_based_f1_score_per_class.append(event_based_class_f1_score)
      segment_based_class_f1_score = self.segment_based_metrics.class_wise_f_measure(class_label)['f_measure']
      segment_based_f1_score_per_class.append(segment_based_class_f1_score)
    average_event_based_f1_score_per_class = np.mean(event_based_f1_score_per_class)
    average_segment_based_f1_score_per_class = np.mean(segment_based_f1_score_per_class)
    return average_event_based_f1_score_per_class, average_segment_based_f1_score_per_class

  def compute_metrics(self, actions):
    window_sizes = []
    for window_size_idx in actions['window_sizes']:
      window_sizes.append(hparams.window_sizes[window_size_idx])
    thresholds = actions['thresholds']
    reference_event_list = self.reference_data[self.file_idx]
    estimated_event_list = self.get_event_list(window_sizes=window_sizes, thresholds=thresholds)
    self.event_based_metrics.evaluate(reference_event_list=reference_event_list, estimated_event_list=estimated_event_list)
    self.segment_based_metrics.evaluate(reference_event_list=reference_event_list, estimated_event_list=estimated_event_list)
    event_based_f1_score, segment_based_f1_score = self.compute_average_class_wise_f_measure()
    return event_based_f1_score, segment_based_f1_score

  def execute(self, actions):
    self.predictions.append(actions)
    # Compute F1-score and Error Rate for file
    event_based_f1_score, segment_based_f1_score = self.compute_metrics(actions)
    # Check if file is last file
    if self.file_idx == (len(self.data) - 1):
      terminal = True
      # If last file, we loop back to the first file
      next_state = dict(state=self.data[0])
      reward = len(self.data) * event_based_f1_score
      print('\nEvent-based F1 = %.3f, Segment-based F1 = %.3f' % (event_based_f1_score, segment_based_f1_score))
      if event_based_f1_score > self.best_event_based_f1_score:
        self.best_event_based_f1_score = event_based_f1_score
        self.best_segment_based_f1_score = segment_based_f1_score
      print('Best event-based F1 = %.3f, Best segment-based F1 = %.3f\n' % (self.best_event_based_f1_score, self.best_segment_based_f1_score))
    else:
      # If not last file, we get the next file
      terminal = False
      next_state = dict(state=self.data[self.file_idx + 1])
      self.file_idx += 1
      reward = 0
    return next_state, terminal, reward