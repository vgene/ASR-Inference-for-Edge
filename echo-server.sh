#! /bin/bash

function cleanup {
	echo 'gracefully exiting...'
	rm echoserver-fifo
}

trap cleanup SIGINT SIGTERM

mknod echoserver-fifo p

# start tcp server
echo 'forwarding 127.0.0.1:12580 to 127.0.0.1:12581'
cat echoserver-fifo | nc -l -k 127.0.0.1 12580 | nc 222.29.98.176 12581 > echoserver-fifo
