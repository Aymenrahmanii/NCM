from ncm_lab.finetune.ncm_loss import target_rows


class FakePre:
    def encode(self, text):
        return [len(w) for w in text.split()]


def test_target_rows_close_with_eos_pad_and_cut():
    rows = target_rows(FakePre(), ["a bb ccc", "a " * 10], length=6, eos=99, pad=0)
    assert rows.tolist()[0] == [1, 2, 3, 99, 0, 0]
    assert rows.tolist()[1] == [1, 1, 1, 1, 1, 99]  # cut to leave room for <eos>
