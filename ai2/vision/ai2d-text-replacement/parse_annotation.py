"""
Some misc functions about annotation
"""

def is_this_text_in_relationship(relationship_annots, text_label, target_relationships):
    for rel in relationship_annots:
        this_rel = relationship_annots[rel]
        if (text_label == this_rel['origin'] or text_label == this_rel['destination']) and this_rel['category'] in target_relationships:
            return True
    return False