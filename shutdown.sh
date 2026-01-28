#!/bin/bash

kill -9 $(ps -ef| grep python| grep -v grep| awk '{print $2}')
kill -9 $(ps -ef| grep VLLM| grep -v grep| awk '{print $2}')
