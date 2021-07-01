import os
import sys
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from collections import defaultdict
import uuid
from tqdm import tqdm
import seaborn as sns
from scipy.stats import chi2

from generate_html import render_html, image_path_to_link

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": ["Palatino"],
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def rename_stereo_method(stereo_method):
  if 'JPLV' in stereo_method:
    stereo_method = 'JPLV Stereo (Normal Rect.)'
  elif 'General' in stereo_method:
    stereo_method = 'JPLV Stereo (General Rect.)'
  return stereo_method

def rename_test_set(test_set_name):
  if test_set_name == 'AirSim Rocks':
    test_set_name = 'AirSim Rocks [No Noise]'
  return test_set_name

#################################################################
#################################################################
#################################################################

def get_subdirectories(basepath):
  x = [d for d in basepath.iterdir() if d.is_dir()]
  return list(sorted(x))


def get_files(basepath):
  x = [f for f in basepath.iterdir() if f.is_file()]
  return list(sorted(x))


def parse_summary_file(filepath):
  stats = defaultdict(lambda: defaultdict(lambda: np.nan))
  if not filepath.exists():
    return stats

  with open(filepath, 'r') as f:
    for line in f:
      parts = line.split(',')
      if len(parts) == 9:
        metric_name = parts[0].strip()
        n, mean, std, lower, q25, q50, q75, upper = [float(p.strip()) for p in parts[1:]]
        stats[metric_name] = {
          'n': int(n),
          'mean': mean,
          'std': std,
          'lower': lower,
          'q25': q25,
          'q50': q50,
          'q75': q75,
          'upper': upper
        }
  return stats

#################################################################
#################################################################
#################################################################

if os.path.exists('./sfm_results'):
  response = input('Delete existing results? [y/n] ')
  if response == 'y':
    if not os.path.exists('/tmp/sfm_results'):
      print('"/tmp/sfm_results" not found!')
      exit()
    shutil.rmtree('./sfm_results')
    shutil.copytree('/tmp/sfm_results', './sfm_results')
else:
  if not os.path.exists('/tmp/sfm_results'):
    print('"/tmp/sfm_results" not found!')
    exit()
  shutil.copytree('/tmp/sfm_results', './sfm_results')

for f in os.listdir('./plots'):
  if f.endswith('.pdf') or f.endswith('.png') or f.endswith('.jpg'):
    os.remove(os.path.join('./plots', f))

basepath = Path('./sfm_results')
stereo_method_dirs = get_subdirectories(basepath)
if len(stereo_method_dirs) == 0:
  print('No stereo methods found!')
  sys.exit(1)

test_set_dirs = []
for stereo_method_dir in stereo_method_dirs:
  test_set_dirs += get_subdirectories(stereo_method_dir)
if len(test_set_dirs) == 0:
  print('No test sets found!')
  sys.exit(1)

test_dirs = []
for test_set_dir in test_set_dirs:
  test_dirs += get_subdirectories(test_set_dir)
if len(test_dirs) == 0:
  print('No tests found!')
  sys.exit(1)

print(f'Found {len(test_dirs)} total test(s)!')

#################################################################
#################################################################
#################################################################

stereo_methods = set()
test_set_names = set()
motions = set()

for test_dir in test_dirs:
  motion = test_dir.parts[-1]
  test_set_name = rename_test_set(test_dir.parts[-2])
  stereo_method = rename_stereo_method(test_dir.parts[-3])

  motions.add(motion)
  test_set_names.add(test_set_name)
  stereo_methods.add(stereo_method)

stereo_methods = list(sorted(stereo_methods))
test_set_names = list(sorted(test_set_names))
motions = list(sorted(motions))

tmp_stats = parse_summary_file(test_dirs[0] / 'stats'/ 'summary.txt')
stat_names = set(tmp_stats.keys())
stat_names.add('Debug stereo')
stat_names.add('Debug depth')
stat_names = list(sorted(stat_names))

#################################################################
#################################################################
#################################################################

def make_dataframe():
  cols = stereo_methods
  rows = pd.MultiIndex.from_product([motions, stat_names])
  df = pd.DataFrame('-', rows, cols)
  return df

#################################################################
#################################################################
#################################################################

def remove_outliers(x, y):
  idx = np.argsort(x)
  x = x[idx]
  y = y[idx]
  q1 = np.quantile(y, 0.25)
  q3 = np.quantile(y, 0.75)
  iqr = q3 - q1
  ymin = q1 - 1.5 * iqr
  ymax = q3 + 1.5 * iqr
  idx = (y >= ymin) & (y <= ymax)
  x = x[idx]
  y = y[idx]
  return x, y

def remove_outliers_df(df, x_name, y_name):
  data = df[[x_name, y_name]].dropna().to_numpy()
  cov = np.cov(data, rowvar=False)
  inv_cov = np.linalg.inv(cov)
  centerpoint = np.mean(data, axis=0)
  distances = []
  for i, val in enumerate(data):
    distance = (val - centerpoint).T.dot(inv_cov).dot(val - centerpoint)
    distances.append(distance)
  cutoff = chi2.ppf(0.90, data.shape[1])
  inlier_idx = np.where(np.array(distances) < cutoff)
  return df.iloc[inlier_idx]

#################################################################
#################################################################
#################################################################

def parse_csv_file(filename):
  if not filename.exists():
    return None
  return pd.read_csv(filename, skipinitialspace=True)

def latex_sanitize(s):
  return s.replace('%', '\%').replace('<=', '$\le$').replace('>=', '$\ge$').replace('<', '$<$').replace('>', '$>$')

#################################################################
#################################################################
#################################################################

palette = 'Set2'
colors = [plt.get_cmap(palette)(1. * i / len(stereo_methods)) for i in range(len(stereo_methods))]
method2color = dict(zip(stereo_methods, colors))

def generate_plot(data, x_name, y_name, title=None):
  df = pd.DataFrame()
  for method, (mean, std) in data.items():
    mean['Algorithm'] = method
    if 'rmse' in y_name.lower():
      df = pd.concat([df, remove_outliers_df(mean, x_name, y_name)])
    else:
      df = pd.concat([df, mean])
  g = sns.FacetGrid(df, col='Algorithm', col_order=stereo_methods, col_wrap=3,
                    hue='Algorithm', hue_order=stereo_methods, palette=palette)
  g.map_dataframe(sns.histplot, x=x_name, y=y_name, stat='probability', bins=(60, 60), common_bins=True)
  g.map_dataframe(sns.kdeplot, x=x_name, y=y_name, cut=0, levels=5, linewidths=0.75)
  g.set_axis_labels(latex_sanitize(x_name), latex_sanitize(y_name))
  g.set_titles("{col_name}")
  if title is not None:
    g.fig.suptitle(title)
    g.fig.tight_layout()
  filename = './plots/%s.png' % str(uuid.uuid1())
  g.savefig(filename, dpi=200)
  plt.close('all')
  return Path(filename)

def generate_boxplot(data, metric, title=None):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
  x, labels, clrs = [], [], []
  for method, statistics in data.items():
    if metric not in statistics:
      plt.close(fig=fig)
      return None

    n = statistics[metric]['n']
    if np.isnan(n) or n == 0:
      continue
    quartiles = [
      statistics[metric]['lower'],
      statistics[metric]['q25'],
      statistics[metric]['q50'],
      statistics[metric]['q75'],
      statistics[metric]['upper']
    ]
    x.append(quartiles)
    label = f'{method}\n[N={n}]'
    labels.append(label.replace(' (', '\n('))
    clrs.append(method2color[method])
  bplot = ax.boxplot(x, labels=labels, whis=[0, 100], patch_artist=True)
  for patch, c in zip(bplot['boxes'], clrs):
    patch.set_facecolor(c)
  plt.setp(bplot['medians'], color='black')
  ax.yaxis.grid(True)
  ax.set_ylabel(latex_sanitize(metric))
  if title is not None:
    ax.set_title(title)
  filename = './plots/%s.png' % str(uuid.uuid1())
  plt.savefig(filename)
  plt.close(fig=fig)
  return Path(filename)

#################################################################
#################################################################
#################################################################

def is_dependent_var(s):
  s = s.lower()
  keywords = ( 'fraction estimated', 'rmse', 'error', 'fraction inliers', 'coverage' )
  for kw in keywords:
    if kw in s:
      return True
  return False

def is_independent_var(s):
  s = s.lower()
  keywords = ( 'baseline - xyz', 'motion angle', 'altitude', 'scene variation' )
  for kw in keywords:
    if kw in s:
      return True
  return False

#################################################################
#################################################################
#################################################################

test_sets = defaultdict(lambda: make_dataframe())  # Test set name --> table of results (dataframe)
csvs = defaultdict(lambda: dict())  # (test set, motion) --> dict from (method name --> csvs)
all_statistics = defaultdict(lambda: dict())  # (test_set, motion) --> dict from (method name --> statistics)
dependent_vars = []
independent_vars = []

for test_dir in tqdm(test_dirs):
  debug_image_files = get_files(test_dir / 'debug')
  depth_image_files = get_files(test_dir / 'depth')

  if len(get_files(test_dir / 'stats')) == 0:
    print(f'No results for "{test_dir}"')
    sys.exit(1)

  stats = parse_summary_file(test_dir / 'stats' / 'summary.txt')

  motion = test_dir.parts[-1]
  test_set_name = rename_test_set(test_dir.parts[-2])
  stereo_method = rename_stereo_method(test_dir.parts[-3])

  df = test_sets[test_set_name]

  for stat_name, statistics in stats.items():
    df[stereo_method][motion][stat_name] = f"{statistics['mean']} ± {statistics['std']}"

  if len(debug_image_files) == 1:
    df[stereo_method][motion]['Debug stereo'] = image_path_to_link(debug_image_files[0])
  if len(depth_image_files) == 1:
    df[stereo_method][motion]['Debug depth'] = image_path_to_link(depth_image_files[0])

  csv_mean = parse_csv_file(test_dir / 'stats' / 'individual_results_mean.csv')
  csv_std = parse_csv_file(test_dir / 'stats' / 'individual_results_std.csv')
  if (csv_mean is not None) and (csv_std is not None) and \
      (len(csv_mean) > 1) and (len(csv_mean) == len(csv_std)):
    csv_mean['Baseline / Altitude'] = csv_mean['Baseline - xyz [m]'] / csv_mean['Altitude [m]']
    csv_mean['Baseline / Altitude'] = csv_mean['Baseline - xyz [m]'] / csv_mean['Altitude [m]']
    csvs[(test_set_name, motion)][stereo_method] = (csv_mean, csv_std)
    dependent_vars = set([c for c in csv_mean.columns if is_dependent_var(c)])
    independent_vars = set([c for c in csv_mean.columns if is_independent_var(c)])
  
  all_statistics[(test_set_name, motion)][stereo_method] = stats

# Generate timeseries plots and add to summary tables
print('Generating plots...')
dependent_vars = list([s.strip() for s in sorted(dependent_vars)])
independent_vars = list([s.strip() for s in sorted(independent_vars)])
for (test_set_name, motion), method2data in csvs.items():
  title = '%s: %s' % (test_set_name, motion)
  df = test_sets[test_set_name]
  for indep_var in independent_vars:
    if indep_var not in df.columns:
      df[indep_var] = '-'
    for dep_var in dependent_vars:
      image_path = generate_plot(method2data, indep_var, dep_var, title=title)
      df[indep_var][motion][dep_var] = image_path_to_link(image_path)
  test_sets[test_set_name] = df

# Generate box-whisker plots and add to summary tables
print('Generating boxplots...')
for (test_set_name, motion), stats_per_method in all_statistics.items():
  title = '%s: %s' % (test_set_name, motion)
  df = test_sets[test_set_name]
  if 'Boxplot' not in df.columns:
    df['Boxplot'] = '-'
  for col in independent_vars + dependent_vars + ['Runtime [ms]']:
    image_path = generate_boxplot(stats_per_method, col, title=title)
    if image_path is not None:
      df['Boxplot'][motion][col] = image_path_to_link(image_path)
  test_sets[test_set_name] = df

# Remove (outer) rows in each dataframe which have no data
print('Removing empty rows...')
all_motions = set([t.parts[-1] for t in test_dirs])
for test_set_name, df in test_sets.items():
  for m in all_motions:
    no_data = (df.loc[m].to_numpy().flatten() == '-').all()
    if no_data:
      df = df.drop(m, level=0)
  test_sets[test_set_name] = df


render_html(test_sets)
print('Done.')
sys.exit(0)
