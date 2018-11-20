function [frames_vec frame_names frame_names_gt] = groundTruthFrames(DATASET)
% DATASET is a string


if strcmp(DATASET,'lobby'),
    frames_vec = [1349
        1353
        1368
        1634
        1649
        2019
        2245
        2247
        2260
        2265
        2388
        2436
        2446
        2457
        2466
        2469
        2497
        2507
        2509
        2514];
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_SwitchLight' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['SwitchLight' num2str(frames_vec(i)) '.bmp'];
    end
    
elseif strcmp(DATASET,'hall'),
    % airport/hall ground truth frames
    frames_vec = [1656
        2180
        2289
        2810
        2823
        2926
        2961
        3049
        3409
        3434
        3800
        3872
        3960
        4048
        4257
        4264
        4333
        4348
        4388
        4432];
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_airport' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['airport' num2str(frames_vec(i)) '.bmp'];
    end
    
elseif strcmp(DATASET, 'bootstrap'),
    
    frames_vec = [1021
        1119
        1285
        1362
        1408
        1416
        1558
        1724
        1832
        1842
        1912
        2238
        2262
        2514
        2624
        2667
        2832
        2880
        2890
        2918];
    
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_b0' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['b0' num2str(frames_vec(i)) '.bmp'];
    end
    
elseif strcmp(DATASET, 'escalator'),
    frames_vec = [2424
        2532
        2678
        2805
        2913
        2952
        2978
        3007
        3078
        3260
        3279
        3353
        3447
        3585
        3743
        4277
        4558
        4595
        4769
        4787];
    
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_Escalator' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['airport' num2str(frames_vec(i)) '.bmp'];
    end
    
    
    
    
elseif strcmp(DATASET, 'campus'),
    
    frames_vec = [1372
        1392
        1394
        1450
        1451
        1489
        1650
        1695
        1698
        1758
        1785
        1812
        1831
        1839
        1845
        2011
        2013
        2029
        2032
        2348];
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_trees' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['trees' num2str(frames_vec(i)) '.bmp'];
    end    
    
    
    
elseif strcmp(DATASET, 'curtain'),
    
    frames_vec = [22772
        22774
        22847
        22849
        22890
        23206
        23222
        23226
        23233
        23242
        23257
        23266
        23786
        23790
        23801
        23817
        23852
        23854
        23857
        23893];
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_Curtain' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['Curtain' num2str(frames_vec(i)) '.bmp'];
    end    
    
    
    
elseif strcmp(DATASET, 'fountain'),
    
    frames_vec = [1157
        1158
        1165
        1179
        1184
        1189
        1190
        1196
        1202
        1204
        1422
        1426
        1430
        1440
        1453
        1465
        1477
        1489
        1494
        1509];
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_Fountain' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['Fountain' num2str(frames_vec(i)) '.bmp'];
    end    
    
    
    
elseif strcmp(DATASET, 'watersurface'),
    
    frames_vec = [1499
        1515
        1523
        1547
        1548
        1553
        1554
        1559
        1575
        1577
        1594
        1597
        1601
        1605
        1615
        1616
        1619
        1620
        1621
        1624];
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_WaterSurface' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['WaterSurface' num2str(frames_vec(i)) '.bmp'];
    end    
    
    
elseif strcmp(DATASET, 'shopping'),
    
    frames_vec = [1433
        1535
        1553
        1581
        1606
        1649
        1672
        1740
        1750
        1761
        1780
        1827
        1862
        1892
        1899
        1920
        1980
        2018
        2055
        2123];
    frame_names_gt=cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names_gt{i} = ['gt_new_ShoppingMall' num2str(frames_vec(i)) '.bmp'];
    end
    frame_names = cell(1,length(frames_vec));
    for i=1:length(frames_vec)
        frame_names{i} = ['ShoppingMall' num2str(frames_vec(i)) '.bmp'];
    end    
    
end


