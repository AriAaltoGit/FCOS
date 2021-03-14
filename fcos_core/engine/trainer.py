# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


# def do_train(
#     model,
#     data_loader,
#     optimizer,
#     scheduler,
#     checkpointer,
#     device,
#     checkpoint_period,
#     arguments,
# ):
#     logger = logging.getLogger("fcos_core.trainer")
#     logger.info("Start training")
#     meters = MetricLogger(delimiter="  ")
#     max_iter = len(data_loader)
    
#     start_iter = arguments["iteration"]
#     model.train()
#     start_training_time = time.time()
#     end = time.time()
#     pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
#     torch.cuda.empty_cache()
#     for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
#         torch.cuda.empty_cache()
#         print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
#         print(torch.cuda.memory_summary())
#         print(torch.cuda.memory_allocated())
#         #torch.cuda.set_per_process_memory_fraction(0.5)

#         data_time = time.time() - end
#         iteration = iteration + 1
#         arguments["iteration"] = iteration

#         # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
#         if not pytorch_1_1_0_or_later:
#             scheduler.step()
#         print(images)
#         print("targets")
#         print(targets)
#         quit()
#         images = images.to(device)
#         targets = [target.to(device) for target in targets]
# #        print(targets[0].bbox)
#         loss_dict = model(images, targets)

#         losses = sum(loss for loss in loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = reduce_loss_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#         meters.update(loss=losses_reduced, **loss_dict_reduced)

#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         if pytorch_1_1_0_or_later:
#             scheduler.step()

#         batch_time = time.time() - end
#         end = time.time()
#         meters.update(time=batch_time, data=data_time)

#         eta_seconds = meters.time.global_avg * (max_iter - iteration)
#         eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

#         if iteration % 20 == 0 or iteration == max_iter:
#             logger.info(
#                 meters.delimiter.join(
#                     [
#                         "eta: {eta}",
#                         "iter: {iter}",
#                         "{meters}",
#                         "lr: {lr:.6f}",
#                         "max mem: {memory:.0f}",
#                     ]
#                 ).format(
#                     eta=eta_string,
#                     iter=iteration,
#                     meters=str(meters),
#                     lr=optimizer.param_groups[0]["lr"],
#                     memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
#                 )
#             )
#         if iteration % checkpoint_period == 0:
#             checkpointer.save("model_{:07d}".format(iteration), **arguments)
#         if iteration == max_iter:
#             checkpointer.save("model_final", **arguments)

    # total_training_time = time.time() - start_training_time
    # total_time_str = str(datetime.timedelta(seconds=total_training_time))
    # logger.info(
    #     "Total training time: {} ({:.4f} s / it)".format(
    #         total_time_str, total_training_time / (max_iter)
    #     )
    # )
