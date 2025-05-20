#!/bin/bash
# 获取脚本所在目录
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
base_dir="$( cd "$script_dir" && pwd )"
echo "base_dir: $base_dir"
cd $base_dir

jobDir=$base_dir
echo "jobDir: $jobDir"
pythonPath=${base_dir}/venv/bin/python

. $base_dir/colors.sh

logName=plant-disease-backend
jobName=plant-disease-backend.py

# 停止已运行的进程
TAILPID=`ps aux | grep "$jobName" | grep -v grep | awk '{print $2}'`
echo "${YELLOW}check $jobName pid $TAILPID ${NOCOLOR}"
if [ "0$TAILPID" != "0" ]; then
    echo "${YELLOW}正在停止 $jobName 进程 (PID: $TAILPID)${NOCOLOR}"
    kill -9 $TAILPID
    echo "${GREEN}$jobName 进程已停止${NOCOLOR}"
else
    echo "${GREEN}没有找到运行中的 $jobName 进程${NOCOLOR}"
fi

# 默认操作为 restart，如果有参数且为 stop，则只执行停止操作
action=${1:-restart}

# 如果操作为 stop，则到此结束
if [ "$action" = "stop" ]; then
    echo "${GREEN}已完成停止操作${NOCOLOR}"
    exit 0
fi

# 否则继续启动服务
mkdir -p logs

echo "${YELLOW}nohup $pythonPath $jobDir/$jobName  > logs/${logName}.log 2>&1 &${NOCOLOR}"
nohup $pythonPath $jobDir/$jobName  > logs/${logName}.log 2>&1 &

echo ------Web后台服务已启动-----

