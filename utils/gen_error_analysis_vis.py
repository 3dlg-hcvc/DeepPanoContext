import os
import json
import random
import pandas as pd
from tqdm import tqdm
from utils.basic_utils import read_pkl
# import html_util

img_path = '/project/3dlg-hcvc/rlsd/www/annotations/docs/final'
tasks_url = 'https://aspis.cmpt.sfu.ca/rlsd/api/scene-manager/tasks/{task_id}'
camera_pose_file = '/project/3dlg-hcvc/rlsd/data/mp3d/equirectangular_camera_poses/{house_id}.jsonl'
rgb_image_path_spec = "https://aspis.cmpt.sfu.ca/projects/rlsd/data/mp3d/equirectangular_rgb_panos/{arch_id}/{panorama_id}.png"


def write_header(f):
    f.write("<html>\n")
    f.write("<head>\n")
    f.write("""<style>
table, td, th {
  border: 1px solid black;
}

table {
  width: 100%;
  border-collapse: collapse;
}
</style>""")
    f.write("</head>\n")
    f.write("<body>\n")


def sample_error_analysis():
    test_tasks = json.load(open("/project/3dlg-hcvc/rlsd/data/psu/rlsd_real_cls25/test.all.json"))
    test_tasks = ['/'.join(t.split('/')[-4:-1]) for t in test_tasks]
    test_tasks = set([t for t in test_tasks if os.path.exists(os.path.join("/local-scratch/qiruiw/research/DeepPanoContext/out/relation_scene_gcn/s3d_25_ro_hs_test_rr_vis/visualization", t, f"est_objs_matched.json"))])
    mirror_tasks = json.load(open("/project/3dlg-hcvc/rlsd/data/annotations/psu_mirror_data.json"))
    mirror_tasks = ['/'.join(t.split('/')[-4:-1]) for t in mirror_tasks]
    mirror_tasks = set([t for t in mirror_tasks if os.path.exists(os.path.join("/local-scratch/qiruiw/research/DeepPanoContext/out/relation_scene_gcn/s3d_25_ro_hs_test_rr_vis/visualization", t, f"est_objs_matched.json"))])
    tasks_wo_mirror = test_tasks.difference(mirror_tasks)

    sample_tasks_wo_mirror = random.sample(list(tasks_wo_mirror), 80)
    sample_tasks_w_mirror = random.sample(list(mirror_tasks), 40)
    sample_tasks = sample_tasks_wo_mirror + sample_tasks_w_mirror

    df = []
    for i, task in tqdm(enumerate(sample_tasks)):
        # import pdb; pdb.set_trace()
        arch_id, pano_id, task_id = task.split('/')
        house_id, level_id = arch_id.split('_')

        gt_data_path = f"/project/3dlg-hcvc/rlsd/data/psu/rr_25_s1_s3d_as/{arch_id}/{pano_id}/{task_id}/gt.pkl"
        est_data_path = f"/project/3dlg-hcvc/rlsd/data/psu/rr_25_s1_s3d_as/{arch_id}/{pano_id}/{task_id}/data.pkl"
        gt_data = read_pkl(gt_data_path)
        est_data = read_pkl(est_data_path)
        num_gt_objs = len(gt_data["objs"])
        num_est_objs = len(est_data["objs"])
        obj_match = json.load(open(os.path.join("/local-scratch/qiruiw/research/DeepPanoContext/out/relation_scene_gcn/s3d_25_ro_hs_test_rr_vis/visualization", arch_id, pano_id, task_id, f"est_objs_matched.json")))
        num_match = len(obj_match["match"])
        num_class_match = len(obj_match["class_match"])

        df.append((task, num_gt_objs, num_est_objs, num_class_match, num_match))
    
    df = pd.DataFrame(df, columns=["task", "num_gt", "num_est", "num_cls_match", "num_match"])
    df.to_csv("/project/3dlg-hcvc/rlsd/data/annotations/error_analysis.csv", index=False)
        

def generate_error_analysis():
    out_path = img_path

    df = pd.read_csv("/project/3dlg-hcvc/rlsd/data/annotations/error_analysis.csv")
    
    with open(os.path.join(out_path, "error_analysis.html"), "w") as f:
        write_header(f)
        f.write("<table>\n")
        columns = ["Task", "RGB", "GT Visual", "Pred Visual", "S3D"]
        f.write("<tr>%s</tr>\n" % ("".join(["<th>%s</th>" % c for c in columns])))
        f.flush()
        
        for i, row in df.iterrows():
            task = row["task"]
            arch_id, pano_id, task_id = task.split('/')
            task_url = f'https://aspis.cmpt.sfu.ca/rlsd/rlsd-scene-editor.html?taskId={task_id}&pixelThreshold=100'
            task_json_url = tasks_url.format(task_id=task_id)
            
            # full_pano_id = task_pano_mapping[task_id]
            house_id, level_id = arch_id.split('_')
            # arch_id = f"{house_id}_{level_id}"
            
            rgb_image = os.path.join("../../data/rlsd_real_cls25", f"{arch_id}/{pano_id}/rgb.png")
            gt_anno_image = os.path.join("../../data/rlsd_real_cls25", f"{arch_id}/{pano_id}/{task_id}/visual.png")
            # gt_layout_render = os.path.join("../../data/rlsd_real_cls25", f"{arch_id}/{pano_id}/{task_id}/scene_mesh.png")
            # layout_bdb3d_render = os.path.join("../../data/rlsd_real", f"{arch_id}/{pano_id}/{task_id}/layout_bdb3d.png")
            
            est_anno_image = os.path.join("s3d_25_ro_hs_test_rr_vis/visualization", arch_id, pano_id, task_id, f"visual.png")
            s3d_layout_render = os.path.join("s3d_25_ro_hs_test_rr_vis/visualization", arch_id, pano_id, task_id, f"layout_bdb3d.png")
            
            f.write("<tr>")
            f.write(f"<td style='width:10%;'>{task_id}<br/>{house_id}_{level_id}<br/>{pano_id}<br/>(<a href='{task_json_url}' target='_blank'>view json</a>)</td>")
            f.write(f"<td style='width:40%;'><img style='width:500px;' src='{rgb_image}'/></td>")
            f.write(f"<td style='width:40%;'><img style='width:500px;' src='{gt_anno_image}'/></td>")
            f.write(f"<td style='width:40%;'><img style='width:500px;' src='{est_anno_image}'/></td>")
            f.write(f"<td style='width:40%;'><img style='width:500px;' src='{s3d_layout_render}'/></td>")
            f.write("</tr>")
            
        # f.write("</tr>")
        f.write("</table>\n")
        f.write("</body>\n")
        f.write("</html>\n")


# sample_error_analysis()
generate_error_analysis()