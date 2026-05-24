@echo off
set PATH=C:\Python\Anaconda\envs\mlagents\Scripts;C:\Python\Anaconda\envs\mlagents;%PATH%
set PYTHONUNBUFFERED=1
set CONDA_PREFIX=C:\Python\Anaconda\envs\mlagents
C:\Python\Anaconda\envs\mlagents\python.exe -u %*
