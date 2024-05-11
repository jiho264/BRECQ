import time


def timer(func):  # 호출할 함수를 매개변수로 받음
    def wrapper(*args):  # 호출할 함수의 매개변수를 받음
        s = time.time()
        time.sleep(1)
        print(func.__name__, "함수 시작")  # __name__으로 함수 이름 출력
        func(args)  # 매개변수로 받은 함수를 호출
        print(func.__name__, "함수 끝")
        print("time :", time.time() - s)

    return wrapper


@timer
def init_quantization_scale(x, channel_wise):
    print("Hello, World!")


init_quantization_scale(1, 1)
