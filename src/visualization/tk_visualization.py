import open3d as o3d

# ## For Pointcloud Visualization
# ## Pointclouds must be in open3d's ocd format.

def vis_sequence(geometries, name="test", capture=False):
    print("n_frames=%s"%len(geometries))
    
    vis=o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    idx=0
    pcd = geometries[idx]
    vis.add_geometry(pcd)
    
    def right_click(vis):
        nonlocal idx
        idx=idx+1
        #print(idx)
        if idx == len(geometries):
            print("end reached, restarting")
            idx = 0
            return
        
        vis.clear_geometries()
        pcd = geometries[idx]

        
        vis.add_geometry(pcd, reset_bounding_box=False)

        ### saves frame screenshot if needed
        if capture:
            vis.capture_screen_image("%s_%s.png"%(name,idx), do_render=True)

    def left_click(vis):
        nonlocal idx
        idx=idx-1
        #print(idx)
        if idx == 0:
            print("start reached, you go reverse")
            idx = len(geometries)-1
            return
        
        vis.clear_geometries()
        pcd = geometries[idx]

        
        vis.add_geometry(pcd, reset_bounding_box=False)

        ### saves frame screenshot if needed
        if capture:
            vis.capture_screen_image("%s_%s.png"%(name,idx), do_render=True)

    def exit_key(vis):
        vis.destroy_window()
        
    # forwad and backward with left and right arrow keys
    vis.register_key_callback(262,right_click) 
    vis.register_key_callback(263,left_click) 
    # exit
    vis.register_key_callback(32,exit_key)
    vis.poll_events()
    vis.run()
