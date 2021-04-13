import traceback

from common.CustomErrors import ValidationError

try:
    for i in range(10):
        try:
            if i == 5:
                i / 0
            else:
                print(i)
        except Exception as e:
            raise ValidationError([1, 2, 3, 4], traceback.format_exc())
except Exception as t:
    print(t)
    print(i)
