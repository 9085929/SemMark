"""
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

示例 1:
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
示例 2:
输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。

提示:
•	1 <= s.length, p.length <= 3 * 104
•	s 和 p 仅包含小写字母

"""
if __name__ == '__main__':
    s = "abca"
    p = "abc"
    s_array = []
    p_array = []
    for c in s:
        s_array.append(c)
    for c in p:
        p_array.append(c)

    for i in range(0, len(s_array) - len(p_array) + 1):
        temp = s_array[i:i + len(p_array)]
        mask = [0] * len(p_array)
        for c in temp:
            for c_p_i in range(0, len(p_array)):
                if p_array[c_p_i] == c:
                    mask[c_p_i] = 1
        flag = 1
        for f in mask:
            flag = flag * f
        if flag == 1:
            print(i)
