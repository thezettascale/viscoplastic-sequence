function step_decay(epoch, lr, step, decay, min_lr)
    return max(lr * decay^(epoch / step), min_lr)
end

function train_epoch(train_state, train_loader, test_loader, loss_fn, model, epoch, cfg::ModelConfig)
    train_loss = 0.0
    test_loss = 0.0

    function objective(model, ps, st, (x, y))
        y_pred, st_new = model((x, y), ps, st)
        return loss_fn(y_pred, y), st_new, (;)
    end

    for (x, y) in train_loader
        _, loss_val, _, train_state = Training.single_train_step!(
            AutoEnzyme(), objective, (x, y), train_state,
        )
        train_loss += loss_val
    end

    st_test = Lux.testmode(train_state.states)
    for (x, y) in test_loader
        y_pred, _ = model((x, y), train_state.parameters, st_test)
        test_loss += loss_fn(y_pred, y)
    end

    new_lr = step_decay(epoch, cfg.learning_rate, cfg.step_rate, cfg.gamma, cfg.min_lr)
    Optimisers.adjust!(train_state.optimizer_state, new_lr)

    n_train = length(train_loader.data)
    n_test = length(test_loader.data)
    return train_state, train_loss / n_train, test_loss / n_test
end
