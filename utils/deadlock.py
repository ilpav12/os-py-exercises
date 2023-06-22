import numpy as np


def calc_safe_sequence(avail, alloc, need, process_names):
    n = len(alloc)     # number of processes
    m = len(alloc[0])  # number of resources
    safe_sequence = []
    work = avail
    finish = [False] * n

    while True:
        print()
        for i in range(n):
            print(f'Process {process_names[i]}', end=' ')
            if finish[i]:
                print('already finished')
                continue

            print('comparing need with work:', need[i], '<=', work)
            if (need[i] > work).any():
                print(f'Process {process_names[i]} has not enough resources')
                continue

            print(f'Process {process_names[i]} completed')
            print(f'work = work + alloc[{process_names[i]}]:', work, '+', alloc[i], '=', np.add(work, alloc[i]))
            work = np.add(work, alloc[i])
            finish[i] = True
            safe_sequence.append(process_names[i])
            break

        if all(finish):
            print()
            print('All processes are finished')
            print('Safe sequence:', safe_sequence)
            return True

        if not any(finish):
            print()
            print('There is no safe sequence')
            return False


def try_grant_request(avail, alloc, need, process, request, process_names):
    if (request > need[process]).any():
        print(f'Request {request} for process {process_names[process]} '
              f'is not granted because it exceeds the maximum resources '
              f'request > need[{process_names[process]}] ({request} > {need[process]})')
        return alloc, need
    print(f'request <= need[{process_names[process]}] ({request} <= {need[process]}) OK')

    if (request > avail).any():
        print(f'Request {request} for process {process_names[process]} '
              f'is not granted because it exceeds the available resources '
              f'request > avail ({request} > {avail})')
        return alloc, need
    print(f'request <= avail ({request} <= {avail}) OK')

    print('avail = avail - request:', avail, '-', request, '=', np.subtract(avail, request))
    new_avail = np.subtract(avail, request)
    print('need = need - request:', need[process], '-', request, '=', np.subtract(need[process], request))
    new_need = need.copy()
    new_need[process] = np.subtract(need[process], request)
    print('alloc[process] = alloc[process] + request:',
          alloc[process], '+', request, '=', np.add(alloc[process], request))
    new_alloc = alloc.copy()
    new_alloc[process] = np.add(alloc[process], request)
    if calc_safe_sequence(new_avail, new_alloc, new_need, process_names):
        print(f'Request {request} for process {process_names[process]} is granted')
    else:
        print(f'Request {request} for process {process_names[process]} is not granted '
              f'because the state would be unsafe')
    return new_alloc, new_need
