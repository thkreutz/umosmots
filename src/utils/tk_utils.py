import time


def log(msg):
    current_time = time.strftime("%D-%H:%M:%S", time.localtime())
    print("%s | %s"% (current_time, msg))

#log("test")
