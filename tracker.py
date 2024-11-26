import math


class Tracker:
    def __init__(self):
        self.center_points = {} # to store the center positions of the objects
        self.id_count = 0 # keep the count of the IDs - each time a new object id detected, count will increase by one

    def update(self, object_rectangles):
        objects_bbs_ids = [] # object bounding boxes and ids

        # get center point of new object
        for rectangle in object_rectangles: 
            x, y, w, h = rectangle
            centre_x = (x + x + w) // 2
            centre_y = (y + y + h) // 2

            # find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                distance = math.hypot(centre_x - pt[0], centre_y - pt[1])

                if distance < 70:
                    self.center_points[id] = (centre_x, centre_y)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # if a new object is detected - assign id to it
            if same_object_detected is False:
                self.center_points[self.id_count] = (centre_x, centre_y)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # remove ids not being used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # update the dictionary
        self.center_points = new_center_points.copy()
        return objects_bbs_ids 