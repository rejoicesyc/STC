import torch
from cc_torch import connected_components_labeling

def make_unss(pred, rat=2):
    assert len(pred.shape) == 4
    assert pred.shape[-1] % 2 == 0 and pred.shape[-2] % 2 == 0
    assert torch.max(pred) == 1.

    ret = torch.zeros_like(pred)

    for i in range(pred.shape[0]):
        pred_2d = pred[i, 0, ...].type(torch.uint8)
        cc_out = connected_components_labeling(pred_2d)
        value_list, value_cnt = torch.unique(cc_out, sorted=True, return_counts=True)
        value_cnt[0] = 0
        sorted_value_cnt, indices = torch.sort(value_cnt, descending=True)

        pred_area = sorted_value_cnt[0]
        for j in range(1, len(value_cnt)):
            t = torch.sum(value_list * (value_cnt == sorted_value_cnt[j - 1]).type(torch.uint8))
            ret[i, 0, cc_out == t] = 1.
            if pred_area > sorted_value_cnt[j] * rat:
                break
            else: 
                pred_area = sorted_value_cnt[j]

    return ret

def test():
    a = torch.tensor([[[
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 1]
    ]]]).cuda()
    print(a.shape)
    print(make_unss(a, rat=1.5))

# test()