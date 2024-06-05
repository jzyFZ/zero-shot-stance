def train(
        model_handler, num_epochs, verbose=True, dev_data=None, early_stopping=False, num_warm=0, is_bert=False
):
    """
    Trains the given model using the given data for the specified
    number of epochs. Prints training loss and evaluation starting
    after 10 epochs. Saves at most 10 checkpoints plus a final one.
    :param model_handler: a holder with a model and data to be trained.
                            Assuming the model is a pytorch model.
    :param num_epochs: the number of epochs to train the model for.
    :param verbose: whether to print train results while training.
                    Default (True): do print intermediate results.
    :param corpus_samplers: list of samplers for individual corpora, None
                            if only evaling on the full corpus.
    :param is_bert: Use BERT as Embedding.
    :param dev_data: test
    :param early_stopping: whether to
    :param num_warm: number of

    """
    prev_dev_loss = 0
    for epoch in range(num_epochs):
        model_handler.train_step()

        if epoch >= num_warm:
            if verbose:
                print("training loss: {}".format(model_handler.loss))
                # eval model on training data
                if not is_bert:
                    # don't do train eval if bert, because super slow
                    trn_scores = model_handler.eval_and_print(data_name='TRAIN')
                # update best scores
                if dev_data is not None:
                    dev_scores = model_handler.eval_and_print(data=dev_data, data_name='DEV')
                    model_handler.save_best(scores=dev_scores)
                else:
                    model_handler.save_best(scores=trn_scores)
            if early_stopping:
                l = model_handler.loss  # will be dev loss because evaled last
                if l < prev_dev_loss:
                    break

    print("TRAINED for {} epochs".format(epoch))

    if early_stopping:
        save_num = "BEST"
    else:
        save_num = "FINAL"

    # save final checkpoint
    model_handler.save(num=save_num)

    # print final training (& dev) scores
    model_handler.eval_and_print(data_name='TRAIN')
    if dev_data is not None:
        model_handler.eval_and_print(data=dev_data, data_name='DEV')
