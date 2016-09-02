"""
Some misc functions about annotation
"""

def is_this_text_in_relationship(relationship_annots, text_label, target_relationships):
    for relationship in relationship_annots:
        if text_label in relationship and relationship_annots[relationship]['category'] in target_relationships:
            return True
    return False