import os
import glob
import numpy as np


def extract_reference_event_list(label_file):
  with open(label_file, 'r') as f:
    lines = f.readlines()
  reference_event_list = []
  for line in lines:
    onset, offset, event_label = line.rstrip('\n').split('\t')
    reference_event_list.append(
      {'onset': float(onset), 'offset': float(offset), 'event_label': event_label}
    )
  return reference_event_list


def load_data(path_to_data, path_to_reference_data, max_length):
  assert os.path.exists(path_to_data), f'"{path_to_data}" does not exist.'
  assert os.path.exists(path_to_reference_data), f'"{path_to_reference_data}" does not exist.'
  data = []
  reference_data = []
  files = glob.glob(os.path.join(path_to_data, '*.npy'))
  for file in files:
    reference_file = os.path.join(path_to_reference_data, os.path.splitext(os.path.basename(file))[0] + '.txt')
    reference_event_list = extract_reference_event_list(reference_file)
    f = np.load(file)
    if len(f) > max_length:
      pf = f[:max_length]
    else:
      padding = ((0, max_length - len(f)), (0, 0))
      pf = np.pad(f, pad_width=padding)
    data.append(pf)
    reference_data.append(reference_event_list)
  return data, reference_data