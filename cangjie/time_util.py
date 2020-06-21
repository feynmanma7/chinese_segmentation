import time


def print_lasts_time(units='s'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()

            last_time = (end_time - start_time)
            print("Lasts ", end='')
            if units == 'ms':
                last_time *= 1000
                print("%.2fms" % last_time)
            else:
                print("%.2fs" % last_time)

            return res

        return wrapper
    return decorator


@print_lasts_time(units='ms')
def test_time(start=1, step=2):
    total = start
    for i in range(10):
        total += step * i
        time.sleep(0.1)
    return total


if __name__ == '__main__':
    total = test_time(start=2, step=3)
    print(total)