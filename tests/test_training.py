# TODO: update tests to train_iterator() when its interface becomes stable
# the code below is for reference

# from fabrique.training import TrainIterator


# def test_train_iter():
#     # limited by max_steps
#     ti = TrainIterator(lambda: iter(range(3)), max_epochs=2, max_steps=5)
#     assert list(ti) == [0, 1, 2, 0, 1]
#     # limited by max_epochs
#     ti = TrainIterator(lambda: iter(range(3)), max_epochs=2, max_steps=100)
#     assert list(ti) == [0, 1, 2, 0, 1, 2]
#     # limited by max_epochs, when max_steps is not specified
#     ti = TrainIterator(lambda: iter(range(3)), max_epochs=2)
#     assert list(ti) == [0, 1, 2, 0, 1, 2]

#     # check step and
#     ti = TrainIterator(lambda: iter(range(3)), max_epochs=2)
#     for _ in range(4):
#         next(ti)
#     assert ti.step == 4
#     assert ti.finished_steps == 4
#     assert ti.epoch == 1
#     assert ti.finished_epochs == 1

#     # construct from a collection
#     ti = TrainIterator([5, 10, 15], max_epochs=2, max_steps=5)
#     assert list(ti) == [5, 10, 15, 5, 10]

#     # batching
#     ti = TrainIterator([5, 10, 15], max_epochs=2, batch_size=2)
#     assert list(ti) == [(5, 10), (15, 5), (10, 15)]
#     ti = TrainIterator([5, 10, 15], max_epochs=3, batch_size=2)
#     assert list(ti) == [(5, 10), (15, 5), (10, 15), (5, 10), (15,)]
#     ti = TrainIterator([5, 10, 15], max_epochs=3, max_steps=4, batch_size=2)
#     assert list(ti) == [(5, 10), (15, 5), (10, 15), (5, 10)]
