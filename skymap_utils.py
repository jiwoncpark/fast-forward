import pickle
import lsst.afw.geom as afw_geom
import numpy as np

def read_skymap(skymap_path):
    with open(skymap_path, "rb") as input_file:
        skymap = pickle.load(input_file)
        #print(skymap.config)
    return skymap

def get_tract_corners(tract_id, skymap_path='/global/cscratch1/sd/desc/DC2/data/Run1.2i/rerun/281118/deepCoadd/skyMap.pickle'):
    '''
    Copied from https://github.com/lsst/pipe_tasks/blob/f3eadeb7a311e131f49332ec787c40cd10c45a47/python/lsst/pipe/tasks/makeDiscreteSkyMap.py#L184
    '''
    skymap = read_skymap(skymap_path)
    tract_info = skymap[tract_id]
    wcs = tract_info.getWcs()
    pos_box = afw_geom.Box2D(tract_info.getBBox())
    pixel_pos_list = (pos_box.getMin(), afw_geom.Point2D(pos_box.getMaxX(), pos_box.getMinY()),
                      pos_box.getMax(), afw_geom.Point2D(pos_box.getMinX(), pos_box.getMaxY()),)
    sky_pos_list = [list(wcs.pixelToSky(pos).getPosition(afw_geom.degrees)) for pos in pixel_pos_list]
    pos_str_list =  ["(%0.3f, %0.3f)" % tuple(sky_pos) for sky_pos in sky_pos_list]
    
    n_patches_x, n_patches_y = tract_info.getNumPatches()
    print("tract %s has corners %s (RA, Dec deg) and %s x %s patches" % (tract_id, ", ".join(pos_str_list),
                                                                         n_patches_x, n_patches_y))
    return np.array(sky_pos_list)
    
    