# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

from functools import lru_cache


# 编辑距离
def edit_distance(s1, s2):
    best_solutions = {}

    @lru_cache(maxsize=2 ** 10)
    def ed(s1, s2, tail=''):

        if len(s1) == 0: return len(s2)
        if len(s2) == 0: return len(s1)

        tail_s1 = s1[-1]
        tail_s2 = s2[-1]

        # use tail to record the removed tail
        candidates = [
            (ed(s1[:-1], s2, tail) + 1, 'DEL {} at position {}'.format(tail_s1, len(s1)), tail),
            # string1 delete tail
            (ed(s1, s2[:-1], tail_s2 + tail) + 1, 'ADD {} at position {}'.format(tail_s2, len(s1) + 1), tail_s2 + tail)
            # string1 add the tail of string2
        ]

        if tail_s1 == tail_s2:
            both_forward = (ed(s1[:-1], s2[:-1], tail_s2 + tail) + 0, '', tail_s2 + tail)
        else:
            both_forward = (ed(s1[:-1], s2[:-1], tail_s2 + tail) + 1,
                            'SUB {} => {} at position {}'.format(tail_s1, tail_s2, len(s1)), tail_s2 + tail)

        candidates.append(both_forward)

        min_distance, operation, tail = min(candidates, key=lambda x: x[0])

        best_solutions[(s1, s2)] = [operation, tail]

        return min_distance

    solution = []

    def parse_solution(s1, s2):
        if (s1, s2) in best_solutions:
            operation, tail = best_solutions[(s1, s2)]
            if operation.startswith('D'):
                solution.append('({:<10}, {}): {}'.format(s1 + tail, s2 + tail, operation))
                return parse_solution(s1[:-1], s2)
            elif operation.startswith('A'):
                solution.append('({:<10}, {}): {}'.format(s1 + tail[1:], s2 + tail[1:], operation))
                return parse_solution(s1, s2[:-1])
            elif operation.startswith('S'):
                solution.append('({:<10}, {}): {}'.format(s1 + tail[1:], s2 + tail[1:], operation))
                return parse_solution(s1[:-1], s2[:-1])
            else:
                return parse_solution(s1[:-1], s2[:-1])

    min_distance = ed(s1, s2)
    parse_solution(s1, s2)
    solution.append('({:<10}, {}): {}'.format(s2, s2, 'Done'))

    return min_distance, solution