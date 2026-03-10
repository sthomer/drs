project(
  'drs',
  ['c', 'fortran'],
  version: '0.1.0',
  default_options: [
    'warning_level=2', 
    'werror=false',
    'python.install_env=auto',
  ]
)

py = import('python').find_installation(pure: false)

subdir('fortran')

py.install_sources(
  [
    'src/drs/__init__.py',
    'src/drs/core.py',
    'src/drs/io.py',
    'src/drs/config.py',
    'src/drs/analysis.py',
    'src/drs/preprocessing.py',
    'src/drs/plotting.py',
  ],
  subdir: 'drs'
)

