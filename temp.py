    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            
            
            m.track_running_stats = True
            m.eval()
            m.train()
            print(m.running_mean)
            _ = m(torch.rand(2, 32, 24, 24))
            print(m.running_mean)
            raise