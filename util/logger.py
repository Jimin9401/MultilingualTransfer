import os
# from torch.utils.tensorboard import SummaryWriter
import numpy as np


# class Logger:
#     def __init__(self, log_dir):
#         """Create a summary writer logging to log_dir."""
#         self.writer = SummaryWriter(log_dir)
#
#     def scalar_summary(self, tag, value, step):
#         """Log a scalar variable."""
#         self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
#
#     def image_summary(self, tag, images, step):
#         """Log a list of images."""
#         for i, img in enumerate(images):
#             # Write the image to a string
#             self.writer.add_image(tag=f'{tag}/{i}', img_tensor=img, global_step=step)
#
#     def histo_summary(self, tag, values, step, bins=1000):
#         """Log a histogram of the tensor of values."""
#         self.writer.add_histogram(tag=tag, values=values, global_step=step, max_bins=bins)


def log_full_eval_test_results_to_file(args, config, results):
    output_eval_file = os.path.join(args.savename, "all_eval_results.txt")

    with open(output_eval_file, "w") as writer:
        writer.write(
            "total batch size=%d\n"
            % (
                    args.per_gpu_train_batch_size
                    * args.gradient_accumulation_step
            )
        )

        writer.write("train num epochs=%d\n" % args.n_epoch)
        writer.write("learning rate=%f\n" % args.lr)
        writer.write("Dataset name=%s\n" % args.dataset)
        writer.write("Model config %s\n" % str(config))

        best_ppl = 10000.0
        best_acc = -9999.99

        for e, result in enumerate(results):
            writer.write("Epoch = %s\n" % (str(e)))
            for key, value in result.items():
                writer.write("%s = %s\n" % (key, str(value)))

                if key == 'eval_ppl':
                    if best_ppl < value:
                        best_ppl = value

                if key == 'eval_loss':
                    if best_acc < value:
                        best_acc = value

            writer.write(
                "-------------------------------------------------------\n")

        writer.write("best ppl : {0}\n best loss : {1}".format(best_ppl, best_acc))


def log_full_test_results_to_file(args, config):
    output_eval_file = args.test_file + ".results.txt"

    with open(output_eval_file, "w") as writer:
        writer.write(
            "total batch size=%d\n"
            % (
                    args.per_gpu_train_batch_size
                    * args.gradient_accumulation_step
            )
        )

        writer.write("number of seeds=%d\n" % len(args.seed_list))
        writer.write("Dataset name=%s\n" % args.dataset)
        writer.write("learning rate=%f\n" % args.lr)
        writer.write("Model config %s\n" % str(config))


        writer.write("average score : %f\n average std : %f" % (float(average_score), float(average_std)))
