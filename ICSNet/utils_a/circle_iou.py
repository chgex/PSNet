


__all__ = ['get_iou']


def get_iou(circleA, circleB):
    """
    compute IoU of two circle area
    :param circleA: tuple: (x,y,r)
    :param circleB: tuple: (x,y,r)
    :return: float: iou which in [0., 1.]
    """
    import math

    x1,y1,r1 = circleA
    x2,y2,r2 = circleB

    if x1 == 0.0 or x2 == 0.0 or y1 == 0.0 or y2 == 0.0 or r1 == 0.0 or r2 == 0.0:
        return 0.

    distance = math.sqrt(math.pow(x1-x2,2) + math.pow(y1-y2,2))
    if distance >= (r1 + r2):
        iou = 0.0
    elif distance <= math.fabs(r1 - r2):
        if r1 < r2:
            inner_area = math.pi * r1 * r1
            outer_area = math.pi * r2 * r2
            iou = inner_area / outer_area
        else:
            inner_area = math.pi * r2 * r2
            outer_area = math.pi * r1 * r1
            iou = inner_area / outer_area
    else:
        angleA = math.acos( (distance**2 + r1**2 - r2**2)/(2*distance*r1) )
        angleB = math.acos( (distance**2 + r2**2 - r1**2)/(2*distance*r2) )

        inter_area = angleA*r1*r1 + angleB*r2*r2 - r1*distance*math.sin(angleA)
        circleA_area = math.pi * r1 * r1
        circleB_area = math.pi * r2 * r2
        union_area = circleA_area + circleB_area - inter_area

        iou = inter_area / (union_area)
    assert iou >= 0.
    return float(iou)



if __name__ == "__main__":

    a = (100, 100, 12)
    b = (100, 100, 12)

    iou = get_iou(a, b)
    print(iou)