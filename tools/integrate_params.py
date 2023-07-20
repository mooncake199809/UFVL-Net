from mmcv.runner import CheckpointLoader, load_checkpoint
import torch
import os

threshold = 0.5
def integrate_multiple_single(sevenscenes_name, sevenscenes_18_channel, save_path, print_keys=False, channel=False, kernel=False):
    assert channel or kernel, print("Type Error!")
    sevenscenes_18_channel_list = []
    new_dict = {}
    specific_name_list = ["downsample.1", "head", "se_layer", "bn"]
    for index, i in enumerate(range(len(sevenscenes_18_channel))):
        model_params = CheckpointLoader.load_checkpoint(sevenscenes_18_channel[index], "cpu")
        sevenscenes_18_channel_list.append(model_params['state_dict'])

    for params_name in sevenscenes_18_channel_list[0].keys():
        if "score" in params_name:
            if channel:
                shared_index = torch.where(sevenscenes_18_channel_list[0][params_name] <= threshold)[0]
                specif_index = torch.where(sevenscenes_18_channel_list[0][params_name] > threshold)[0]
                new_dict.update({
                    params_name[:params_name.rfind(".")] + ".weight" : sevenscenes_18_channel_list[0][params_name[:params_name.rfind(".")] + ".weight"][:,shared_index,:,:],
                    params_name : sevenscenes_18_channel_list[0][params_name]
                })
                for index, scene_name in enumerate(sevenscenes_name):
                    new_dict.update({
                        scene_name + "_" + params_name[:params_name.rfind(".")] + ".specific_weight" : sevenscenes_18_channel_list[index][params_name[:params_name.rfind(".")] + ".specific_weight"][:,specif_index,:,:],
                    })
            elif kernel:
                shared_index = torch.where(sevenscenes_18_channel_list[0][params_name] <= threshold)
                specif_index = torch.where(sevenscenes_18_channel_list[0][params_name] > threshold)
                new_dict.update({
                    params_name[:params_name.rfind(".")] + ".weight" : sevenscenes_18_channel_list[0][params_name[:params_name.rfind(".")] + ".weight"][:,:,shared_index[0],shared_index[1]],
                    params_name : sevenscenes_18_channel_list[0][params_name]
                })
                for index, scene_name in enumerate(sevenscenes_name):
                    new_dict.update({
                        scene_name + "_" + params_name[:params_name.rfind(".")] + ".specific_weight" : sevenscenes_18_channel_list[index][params_name[:params_name.rfind(".")] + ".specific_weight"][:,:,specif_index[0],specif_index[1]],
                    })


        elif any(specific_name in params_name for specific_name in specific_name_list):
            for index, scene_name in enumerate(sevenscenes_name):
                new_dict.update({
                    scene_name + "_" + params_name : sevenscenes_18_channel_list[index][params_name]
                })
        elif "weight" not in params_name:
            print("ERROR")
            exit()
    torch.save(new_dict, save_path)



def print_checkpoints(save_path):
    model_params = torch.load(save_path, map_location="cpu")
    if "state_dict" in model_params.keys():
        model_params_name = model_params["state_dict"]
    else:
        model_params_name = model_params
    # for i in model_params_name.keys():
    #     print(i)
    print(model_params_name.keys())
    print(model_params_name["backbone.layer1.1.conv1.score"].shape)



scene_type = "12Scenes"

if scene_type == "7Scenes":
    integrate_multiple_single(sevenscenes_name = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
                            sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18channel/iter_200000_chess.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18channel/iter_200000_fire.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18channel/iter_200000_heads.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18channel/iter_200000_office.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18channel/iter_200000_pumpkin.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18channel/iter_200000_redkitchen.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18channel/iter_200000_stairs.pth"],
                            save_path = "/home/dk/ufvl_net/weights/18_channel_7scenes.pth",
                            print_keys=False, channel=True, kernel=False)

    integrate_multiple_single(sevenscenes_name = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
                            sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18kernel/iter_200000_chess.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18kernel/iter_200000_fire.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18kernel/iter_200000_heads.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18kernel/iter_200000_office.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18kernel/iter_200000_pumpkin.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18kernel/iter_200000_redkitchen.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet18kernel/iter_200000_stairs.pth"],
                            save_path = "/home/dk/ufvl_net/weights/18_kernel_7scenes.pth",
                            print_keys=False, channel=False, kernel=True)

    integrate_multiple_single(sevenscenes_name = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
                            sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34channel/iter_200000_channel_chess_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34channel/iter_200000_channel_fire_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34channel/iter_200000_channel_heads_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34channel/iter_200000_channel_office_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34channel/iter_200000_channel_pumpkin_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34channel/iter_200000_channel_redkitchen_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34channel/iter_200000_channel_stairs_34.pth"],
                            save_path = "/home/dk/ufvl_net/weights/34_channel_7scenes.pth",
                            print_keys=False, channel=True, kernel=False)


    integrate_multiple_single(sevenscenes_name = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
                            sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34kernel/iter_200000_kernel_chess_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34kernel/iter_200000_kernel_fire_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34kernel/iter_200000_kernel_heads_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34kernel/iter_200000_kernel_office_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34kernel/iter_200000_kernel_pumpkin_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34kernel/iter_200000_kernel_redkitchen_34.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet34kernel/iter_200000_kernel_stairs_34.pth"],
                            save_path = "/home/dk/ufvl_net/weights/34_kernel_7scenes.pth",
                            print_keys=False, channel=False, kernel=True)


    integrate_multiple_single(sevenscenes_name = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
                            sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50channel/iter_200000_channel_chess.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50channel/iter_200000_channel_fire.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50channel/iter_200000_channel_heads.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50channel/iter_200000_channel_office.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50channel/iter_200000_channel_pumpkin.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50channel/iter_200000_channel_redkitchen.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50channel/iter_200000_channel_stairs.pth"],
                            save_path = "/home/dk/ufvl_net/weights/50_channel_7scenes.pth",
                            print_keys=False, channel=True, kernel=False)


    integrate_multiple_single(sevenscenes_name = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
                            sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50kernel/iter_200000_kernel_chess_50.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50kernel/iter_200000_kernel_fire_50.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50kernel/iter_200000_kernel_heads_50.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50kernel/iter_200000_kernel_office_50.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50kernel/iter_200000_kernel_pumpkin_50.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50kernel/iter_200000_kernel_redkitchen_50.pth",
                                                        "/mnt/share/sda-8T/dk/OFVM_MS2.0/resnet50kernel/iter_200000_kernel_stairs_50.pth"],
                            save_path = "/home/dk/ufvl_net/weights/50_kernel_7scenes.pth",
                            print_keys=False, channel=False, kernel=True)
elif scene_type == "12Scenes":
    integrate_multiple_single(sevenscenes_name = ["apt1kitchen", "apt1living", "apt2bed", "apt2kitchen", "apt2living", "apt2luke", "office1gates362",
                                                  "office1gates381", "office1lounge", "office1manolis", "office25a", "office25b"],
                            sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_apt1_kitchen/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_apt1_living/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_apt2_bed/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_apt2_kitchen/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_apt2_living/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_apt2_luke/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_office1_gates362/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_office1_gates381/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_office1_lounge/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_office1_manolis/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_office2_5a/iter_200000.pth",
                                                      "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_channel_wise/_task_office2_5b/iter_200000.pth"],
                            save_path = "/home/dk/ufvl_net/weights/18_channel_12scenes.pth",
                            print_keys=False, channel=True, kernel=False)
    
    integrate_multiple_single(sevenscenes_name = ["apt1kitchen", "apt1living", "apt2bed", "apt2kitchen", "apt2living", "apt2luke", "office1gates362",
                                                "office1gates381", "office1lounge", "office1manolis", "office25a", "office25b"],
                        sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_apt1_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_apt1_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_apt2_bed/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_apt2_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_apt2_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_apt2_luke/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_office1_gates362/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_office1_gates381/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_office1_lounge/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_office1_manolis/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_office2_5a/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_18_kernel_wise/_task_office2_5b/iter_200000.pth"],
                        save_path = "/home/dk/ufvl_net/weights/18_kernel_12scenes.pth",
                        print_keys=False, channel=False, kernel=True)

    integrate_multiple_single(sevenscenes_name = ["apt1kitchen", "apt1living", "apt2bed", "apt2kitchen", "apt2living", "apt2luke", "office1gates362",
                                                "office1gates381", "office1lounge", "office1manolis", "office25a", "office25b"],
                        sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_apt1_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_apt1_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_apt2_bed/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_apt2_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_apt2_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_apt2_luke/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_office1_gates362/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_office1_gates381/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_office1_lounge/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_office1_manolis/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_office2_5a/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_channel_wise/_task_office2_5b/iter_200000.pth"],
                        save_path = "/home/dk/ufvl_net/weights/34_channel_12scenes.pth",
                        print_keys=False, channel=True, kernel=False)
    
    integrate_multiple_single(sevenscenes_name = ["apt1kitchen", "apt1living", "apt2bed", "apt2kitchen", "apt2living", "apt2luke", "office1gates362",
                                                "office1gates381", "office1lounge", "office1manolis", "office25a", "office25b"],
                        sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_apt1_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_apt1_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_apt2_bed/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_apt2_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_apt2_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_apt2_luke/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_office1_gates362/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_office1_gates381/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_office1_lounge/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_office1_manolis/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_office2_5a/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_34_kernel_wise/_task_office2_5b/iter_200000.pth"],
                        save_path = "/home/dk/ufvl_net/weights/34_kernel_12scenes.pth",
                        print_keys=False, channel=False, kernel=True)
    
    integrate_multiple_single(sevenscenes_name = ["apt1kitchen", "apt1living", "apt2bed", "apt2kitchen", "apt2living", "apt2luke", "office1gates362",
                                                "office1gates381", "office1lounge", "office1manolis", "office25a", "office25b"],
                        sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_apt1_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_apt1_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_apt2_bed/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_apt2_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_apt2_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_apt2_luke/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_office1_gates362/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_office1_gates381/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_office1_lounge/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_office1_manolis/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_office2_5a/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_channel_wise/_task_office2_5b/iter_200000.pth"],
                        save_path = "/home/dk/ufvl_net/weights/50_channel_12scenes.pth",
                        print_keys=False, channel=True, kernel=False)
    
    integrate_multiple_single(sevenscenes_name = ["apt1kitchen", "apt1living", "apt2bed", "apt2kitchen", "apt2living", "apt2luke", "office1gates362",
                                                "office1gates381", "office1lounge", "office1manolis", "office25a", "office25b"],
                        sevenscenes_18_channel = ["/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_apt1_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_apt1_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_apt2_bed/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_apt2_kitchen/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_apt2_living/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_apt2_luke/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_office1_gates362/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_office1_gates381/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_office1_lounge/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_office1_manolis/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_office2_5a/iter_200000.pth",
                                                    "/mnt/share/sda-8T/dk/OFVM_MS2.0/12Scenes/ofvl_ms_2_50_kernel_wise/_task_office2_5b/iter_200000.pth"],
                        save_path = "/home/dk/ufvl_net/weights/50_kernel_12scenes.pth",
                        print_keys=False, channel=False, kernel=True)
    

    

    
# print_checkpoints("/mnt/share/sda-8T/dk/12Scenes/XT/res18_2/best_median_381_211500.pth")




